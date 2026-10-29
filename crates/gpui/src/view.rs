use crate::{
    AnyElement, AnyEntity, AnyWeakEntity, App, AvailableSpace, Bounds, ContentMask, Context,
    Element, ElementId, Entity, EntityId, GlobalElementId, InspectorElementId, IntoElement,
    LayoutId, PaintIndex, Pixels, PrepaintStateIndex, Render, RenderOnce, Size, Style,
    StyleRefinement, TextStyle, TransformationMatrix, WeakEntity,
};
use crate::{Empty, Window};
use anyhow::Result;
use collections::FxHashSet;
use refineable::Refineable;
use std::mem;
use std::rc::Rc;
use std::{
    any::{TypeId, type_name},
    fmt,
    ops::Range,
};

/// A dynamically-typed view handle that can be downcast to a specific `Entity<V>`.
///
/// This is the type-erased counterpart to [`ViewElement`]: it holds an entity plus
/// a function pointer to its render, and is itself a [`View`], so embedding it as an
/// element goes through the same [`ViewElement`] machinery as any other view.
#[derive(Clone, Debug)]
pub struct AnyView {
    entity: AnyEntity,
    render: fn(&AnyView, &mut Window, &mut App) -> AnyElement,
    view_type_name: &'static str,
}

impl<V: Render> From<Entity<V>> for AnyView {
    fn from(value: Entity<V>) -> Self {
        AnyView {
            entity: value.into_any(),
            render: any_view::render::<V>,
            view_type_name: type_name::<V>(),
        }
    }
}

impl AnyView {
    /// Embed this view as a cached [`ViewElement`] laid out at `style`.
    ///
    /// The rendered subtree is recycled from the previous frame unless
    /// [Context::notify] was called on the backing entity since it was rendered
    /// (or [Window::refresh] is called, which ignores caching).
    pub fn cached(self, style: StyleRefinement) -> ViewElement<AnyView> {
        self.into_element().cached(style)
    }

    /// Embed this view cached and isolated: [`Context::notify`] on its entity
    /// draws it again alone, inside the frames its cached ancestors reuse. See
    /// [`ViewElement::isolated`].
    pub fn isolated(self, style: StyleRefinement) -> ViewElement<AnyView> {
        let view = self.clone();
        self.into_element()
            .cached(style)
            .isolated(Rc::new(move |window, cx| {
                render_view(view.clone(), window, cx)
            }))
    }

    /// Convert this to a weak handle.
    pub fn downgrade(&self) -> AnyWeakView {
        AnyWeakView {
            entity: self.entity.downgrade(),
            render: self.render,
            view_type_name: self.view_type_name,
        }
    }

    /// Convert this to a [Entity] of a specific type.
    /// If this handle does not contain a view of the specified type, returns itself in an `Err` variant.
    pub fn downcast<T: 'static>(self) -> Result<Entity<T>, Self> {
        match self.entity.downcast() {
            Ok(entity) => Ok(entity),
            Err(entity) => Err(Self {
                entity,
                render: self.render,
                view_type_name: self.view_type_name,
            }),
        }
    }

    /// Gets the [TypeId] of the underlying view.
    pub fn entity_type(&self) -> TypeId {
        self.entity.entity_type
    }

    /// The [`EntityId`] of this view.
    pub fn entity_id(&self) -> EntityId {
        self.entity.entity_id()
    }
}

impl PartialEq for AnyView {
    fn eq(&self, other: &Self) -> bool {
        self.entity == other.entity
    }
}

impl Eq for AnyView {}

/// `AnyView` is the type-erased [`View`]: its `render` is a function pointer rather
/// than a concrete type, but it participates in the reactive graph exactly like any
/// other view via [`ViewElement`].
impl View for AnyView {
    fn entity_id(&self) -> Option<EntityId> {
        Some(self.entity.entity_id())
    }

    #[cfg(not(gpui_hot_reload))]
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        (self.render)(&self, window, cx)
    }

    // The stored function pointer was captured from the original binary, so it must be looked up
    // in the jump table to reach the patched render.
    #[cfg(gpui_hot_reload)]
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        subsecond::HotFn::current(self.render).call((&self, window, cx))
    }
}

impl<V: 'static + Render> IntoElement for Entity<V> {
    type Element = ViewElement<Entity<V>>;

    fn into_element(self) -> Self::Element {
        ViewElement::with_type_name(self, type_name::<V>())
    }

    #[inline(never)]
    fn into_any_element(self) -> AnyElement {
        self.into_element().into_any()
    }
}

impl IntoElement for AnyView {
    type Element = ViewElement<AnyView>;

    fn into_element(self) -> Self::Element {
        let view_type_name = self.view_type_name;
        ViewElement::with_type_name(self, view_type_name)
    }
}

/// A weak, dynamically-typed view handle.
pub struct AnyWeakView {
    entity: AnyWeakEntity,
    render: fn(&AnyView, &mut Window, &mut App) -> AnyElement,
    view_type_name: &'static str,
}

impl AnyWeakView {
    /// Upgrade to a strong `AnyView` handle, if the view is still alive.
    pub fn upgrade(&self) -> Option<AnyView> {
        let entity = self.entity.upgrade()?;
        Some(AnyView {
            entity,
            render: self.render,
            view_type_name: self.view_type_name,
        })
    }
}

impl<V: 'static + Render> From<WeakEntity<V>> for AnyWeakView {
    fn from(view: WeakEntity<V>) -> Self {
        AnyWeakView {
            entity: view.into(),
            render: any_view::render::<V>,
            view_type_name: type_name::<V>(),
        }
    }
}

impl PartialEq for AnyWeakView {
    fn eq(&self, other: &Self) -> bool {
        self.entity == other.entity
    }
}

impl std::fmt::Debug for AnyWeakView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnyWeakView")
            .field("entity_id", &self.entity.entity_id)
            .finish_non_exhaustive()
    }
}

mod any_view {
    use crate::{AnyElement, AnyView, App, IntoElement, Render, Window};

    pub(crate) fn render<V: 'static + Render>(
        view: &AnyView,
        window: &mut Window,
        cx: &mut App,
    ) -> AnyElement {
        let view = view.clone().downcast::<V>().unwrap();
        // Record the view's Render type name so the accessibility debug dump can
        // attribute nodes to the view that produced them.
        #[cfg(debug_assertions)]
        window
            .a11y
            .view_type_names
            .insert(view.entity_id(), std::any::type_name::<V>());
        view.update(cx, |view, cx| view.render(window, cx).into_any_element())
    }
}

/// A renderable that participates in GPUI's reactive graph — the unifying model
/// behind [`Render`] and [`RenderOnce`].
///
/// When `entity_id()` returns `Some`, that id becomes the view's identity: it gets
/// a unique element-id space (so internal `use_state` / `.id(..)` never collide
/// across siblings) and `cx.notify()` on that entity re-renders only this view's
/// subtree. `None` behaves like a stateless component.
///
/// You rarely implement `View` directly. `Entity<T: Render>` and any `T: RenderOnce`
/// get a blanket impl below; implement it by hand only when a component needs both
/// parent-supplied props *and* a backing entity for identity.
pub trait View: 'static + Sized {
    /// This view's identity, if it has one. A view typically holds the backing
    /// entity as a field and returns its [`EntityId`] here.
    ///
    /// The id becomes this view's [`ElementId`], so two views keyed on the same
    /// entity must not be rendered at the same position in the element tree
    /// (e.g. as siblings under the same parent): their internal element state
    /// (`use_state`, scroll offsets, etc.) would silently collide. Nesting is
    /// fine — the id is scoped by the parent path.
    fn entity_id(&self) -> Option<EntityId>;

    /// Render this view into an element tree, consuming `self`.
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement;
}

/// A stateless component (`RenderOnce`) is a `View` with no identity.
impl<T: RenderOnce> View for T {
    fn entity_id(&self) -> Option<EntityId> {
        None
    }

    #[inline]
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        RenderOnce::render(self, window, cx)
    }
}

/// An entity that renders itself (`Render`) is a `View` keyed on its own id.
impl<T: Render> View for Entity<T> {
    fn entity_id(&self) -> Option<EntityId> {
        Some(Entity::entity_id(self))
    }

    #[inline]
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        self.update(cx, |this, cx| {
            Render::render(this, window, cx).into_any_element()
        })
    }
}

impl<T: Render> Entity<T> {
    /// Embed this entity as a cached [`ViewElement`] laid out at `style`.
    ///
    /// The rendered subtree is reused until the entity is notified (or the
    /// cached bounds / text style change). Caching requires a definite size:
    /// a cached view is laid out from `style` and is *not* measured from its
    /// contents. Use [`ViewElement::new`] (or `.child(entity)`) for the
    /// uncached case.
    #[track_caller]
    pub fn cached(self, style: StyleRefinement) -> ViewElement<Entity<T>> {
        ViewElement::with_type_name(self, type_name::<T>()).cached(style)
    }

    /// Embed this entity cached and isolated: [`Context::notify`] on it draws
    /// it again alone, inside the frames its cached ancestors reuse. See
    /// [`ViewElement::isolated`].
    #[track_caller]
    pub fn isolated(self, style: StyleRefinement) -> ViewElement<Entity<T>> {
        let view = self.clone();
        ViewElement::with_type_name(self, type_name::<T>())
            .cached(style)
            .isolated(Rc::new(move |window, cx| {
                render_view(view.clone(), window, cx)
            }))
    }
}

/// The element type for [`View`] implementations. Wraps a `View` and hooks it
/// into layout, prepaint, and paint. Constructed via [`ViewElement::new`].
#[doc(hidden)]
pub struct ViewElement<V: View> {
    view: Option<V>,
    entity_id: Option<EntityId>,
    cached_style: Option<StyleRefinement>,
    /// Renders the view again on its own (see [`ViewElement::isolated`]).
    island: Option<Rc<dyn Fn(&mut Window, &mut App) -> AnyElement>>,
    #[cfg(feature = "profiler")]
    view_type_name: &'static str,
    #[cfg(debug_assertions)]
    source: &'static core::panic::Location<'static>,
}

impl<V: View> ViewElement<V> {
    /// Wrap a [`View`] as an element.
    #[track_caller]
    pub fn new(view: V) -> Self {
        Self::with_type_name(view, type_name::<V>())
    }

    /// Wraps a view whose draw time profiling attributes to `view_type_name`,
    /// such as an entity's type rather than [`Entity`]'s.
    #[track_caller]
    #[cfg_attr(not(feature = "profiler"), expect(unused_variables))]
    pub(crate) fn with_type_name(view: V, view_type_name: &'static str) -> Self {
        let entity_id = view.entity_id();
        ViewElement {
            entity_id,
            cached_style: None,
            island: None,
            #[cfg(feature = "profiler")]
            view_type_name,
            view: Some(view),
            #[cfg(debug_assertions)]
            source: core::panic::Location::caller(),
        }
    }

    /// Enable caching of this view's rendered subtree, laid out at `style`.
    /// The composer supplies the layout style because caching skips rendering
    /// the contents to measure them.
    ///
    /// Crate-private on purpose: caching is only sound for entity-backed views,
    /// where [`Context::notify`] is the contract that busts the cache. A stateless
    /// view has no such contract, so a frozen subtree could never be invalidated.
    /// Reach this through [`Entity::cached`] or [`AnyView::cached`], which are
    /// entity-backed by construction.
    pub(crate) fn cached(mut self, style: StyleRefinement) -> Self {
        self.cached_style = Some(style);
        self
    }

    /// Isolate this cached view: a [`Context::notify`] on its entity marks
    /// only it dirty, not its ancestors. Its cached ancestors reuse their
    /// previous frames around it and draw it again alone, in the bounds,
    /// content mask, transformation, opacity and text style it was last drawn
    /// with, at the same place in paint order, hit testing and the dispatch
    /// tree. So its layout box must not depend on its contents (it is laid out
    /// from the cached style); a change that moves or resizes it notifies an
    /// ancestor, which draws it again as usual. Inside a deferred draw it is
    /// not isolated. `render` renders the view again on its own.
    ///
    /// What it paints outside its box (overflow, shadows, deferred children
    /// excepted) is clipped by the content mask it was last drawn with and
    /// placed in paint order where it was: an ancestor that would clip or
    /// order it differently draws it again when it renders.
    ///
    /// Views that redraw often inside a large, mostly static tree (a canvas
    /// inside application chrome) use it so their frames do not rebuild the
    /// tree around them; with [`Window::set_root_cached`] the root is reused
    /// too.
    pub(crate) fn isolated(
        mut self,
        render: Rc<dyn Fn(&mut Window, &mut App) -> AnyElement>,
    ) -> Self {
        self.island = Some(render);
        self
    }
}

/// An isolated view drawn again inside a reused cached ancestor (see
/// [`ViewElement::isolated`]): where it was, its dispatch node now, and what
/// it rendered.
pub(crate) struct Splice {
    island: crate::window::Island,
    node: Option<crate::DispatchNodeId>,
    element: Option<AnyElement>,
}

/// What a view element's prepaint leaves for its paint: the rendered
/// element (drawn), or the isolated views drawn again in a reused frame.
#[doc(hidden)]
pub struct ViewPrepaint {
    element: Option<AnyElement>,
    splices: Vec<Splice>,
}

impl<V: View> IntoElement for ViewElement<V> {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

struct ViewElementState {
    prepaint_range: Range<PrepaintStateIndex>,
    paint_range: Range<PaintIndex>,
    cache_key: ViewElementCacheKey,
    accessed_entities: FxHashSet<EntityId>,
}

struct ViewElementCacheKey {
    bounds: Bounds<Pixels>,
    content_mask: ContentMask<Pixels>,
    text_style: TextStyle,
    // Hitboxes and mouse listeners capture it, so a cached view reuses them only under the
    // same one.
    element_transformation: TransformationMatrix,
}

#[cfg(not(gpui_hot_reload))]
#[inline]
fn render_view<V: View>(view: V, window: &mut Window, cx: &mut App) -> AnyElement {
    view.render(window, cx).into_any_element()
}

/// Every view and component render passes through this hot-patchable boundary, so the
/// patched code for that render and everything it calls runs on the next frame.
#[cfg(gpui_hot_reload)]
fn render_view<V: View>(view: V, window: &mut Window, cx: &mut App) -> AnyElement {
    #[inline(never)]
    fn render_view_impl<V: View>(view: V, window: &mut Window, cx: &mut App) -> AnyElement {
        view.render(window, cx).into_any_element()
    }

    subsecond::HotFn::current(render_view_impl::<V>).call((view, window, cx))
}

impl<V: View> Element for ViewElement<V> {
    type RequestLayoutState = Option<AnyElement>;
    type PrepaintState = ViewPrepaint;

    fn id(&self) -> Option<ElementId> {
        self.entity_id.map(ElementId::View)
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        #[cfg(debug_assertions)]
        return Some(self.source);

        #[cfg(not(debug_assertions))]
        return None;
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        if let Some(entity_id) = self.entity_id {
            // Stateful path: create a reactive boundary.
            #[cfg(feature = "profiler")]
            let start = window.window_profiler.begin_view();
            let view = &mut self.view;
            let layout = request_layout_view(
                entity_id,
                self.cached_style.as_ref(),
                window,
                cx,
                &mut |window, cx| render_view(view.take().unwrap(), window, cx),
            );
            #[cfg(feature = "profiler")]
            window.window_profiler.end_view(
                start,
                self.view_type_name,
                crate::profiler::ViewPhase::RequestLayout,
            );
            layout
        } else {
            // Stateless path: isolate subtree via type name (no entity identity).
            request_layout_component(type_name::<V>(), window, cx, &mut |window, cx| {
                render_view(self.view.take().unwrap(), window, cx)
            })
        }
    }

    fn prepaint(
        &mut self,
        global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        element: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) -> ViewPrepaint {
        if let Some(entity_id) = self.entity_id {
            // Stateful path.
            #[cfg(feature = "profiler")]
            let start = window.window_profiler.begin_view();
            let prepainted = prepaint_view(
                entity_id,
                global_id,
                bounds,
                element,
                self.island.as_ref(),
                window,
                cx,
                &mut |window, cx| render_view(self.view.take().unwrap(), window, cx),
            );
            #[cfg(feature = "profiler")]
            window.window_profiler.end_view(
                start,
                self.view_type_name,
                crate::profiler::ViewPhase::Prepaint,
            );
            prepainted
        } else {
            // Stateless path: just prepaint the element.
            ViewPrepaint {
                element: prepaint_component(type_name::<V>(), element, window, cx),
                splices: Vec::new(),
            }
        }
    }

    fn paint(
        &mut self,
        global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        element: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        if let Some(entity_id) = self.entity_id {
            // Stateful path.
            #[cfg(feature = "profiler")]
            let start = window.window_profiler.begin_view();
            paint_view(
                entity_id,
                self.cached_style.is_some(),
                self.island.is_some(),
                global_id,
                element,
                window,
                cx,
            );
            #[cfg(feature = "profiler")]
            window.window_profiler.end_view(
                start,
                self.view_type_name,
                crate::profiler::ViewPhase::Paint,
            );
        } else {
            // Stateless path: just paint the element.
            paint_component(std::any::type_name::<V>(), &mut element.element, window, cx);
        }
    }
}

/// A view that renders nothing
pub struct EmptyView;

impl Render for EmptyView {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        Empty
    }
}

#[inline(never)]
fn request_layout_view(
    entity_id: EntityId,
    cached_style: Option<&StyleRefinement>,
    window: &mut Window,
    cx: &mut App,
    render: &mut dyn FnMut(&mut Window, &mut App) -> AnyElement,
) -> (LayoutId, Option<AnyElement>) {
    window.with_rendered_view(entity_id, |window| {
        let caching_disabled = window.is_inspector_picking(cx);
        match cached_style {
            Some(style) if !caching_disabled => {
                let mut root_style = Style::default();
                root_style.refine(style);
                let layout_id = window.request_layout(root_style, None, cx);
                (layout_id, None)
            }
            _ => {
                let mut element = render(window, cx);
                let layout_id = element.request_layout(window, cx);
                (layout_id, Some(element))
            }
        }
    })
}

#[inline(never)]
fn request_layout_component(
    name: &'static str,
    window: &mut Window,
    cx: &mut App,
    render: &mut dyn FnMut(&mut Window, &mut App) -> AnyElement,
) -> (LayoutId, Option<AnyElement>) {
    window.with_id(ElementId::from(name), |window| {
        let mut element = render(window, cx);
        let layout_id = element.request_layout(window, cx);
        (layout_id, Some(element))
    })
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn prepaint_view(
    entity_id: EntityId,
    global_id: Option<&GlobalElementId>,
    bounds: Bounds<Pixels>,
    element: &mut Option<AnyElement>,
    island: Option<&Rc<dyn Fn(&mut Window, &mut App) -> AnyElement>>,
    window: &mut Window,
    cx: &mut App,
    render: &mut dyn FnMut(&mut Window, &mut App) -> AnyElement,
) -> ViewPrepaint {
    window.set_view_id(entity_id);
    window.with_rendered_view(entity_id, |window| {
        if let Some(mut element) = element.take() {
            let prepaint_start = window.prepaint_index();
            element.prepaint(window, cx);
            if let Some(render) = island {
                let range = prepaint_start..window.prepaint_index();
                let record = window.island_here(
                    entity_id,
                    render.clone(),
                    global_id.unwrap().clone(),
                    bounds,
                    range,
                );
                window.push_island(record);
            }
            return ViewPrepaint {
                element: Some(element),
                splices: Vec::new(),
            };
        }

        let (element, splices, range) = window.with_element_state::<ViewElementState, _>(
            global_id.unwrap(),
            |element_state, window| {
                let content_mask = window.content_mask();
                let text_style = window.text_style();
                let element_transformation = window.element_transformation();

                if let Some(mut element_state) = element_state
                    && element_state.cache_key.bounds == bounds
                    && element_state.cache_key.content_mask == content_mask
                    && element_state.cache_key.text_style == text_style
                    && element_state.cache_key.element_transformation == element_transformation
                    && !window.dirty_views.contains(&entity_id)
                    && !window.refreshing
                {
                    let prepaint_start = window.prepaint_index();
                    let splices = reuse_prepaint_around_islands(
                        element_state.prepaint_range.clone(),
                        window,
                        cx,
                    );
                    cx.entities
                        .extend_accessed(&element_state.accessed_entities);
                    let prepaint_end = window.prepaint_index();
                    element_state.prepaint_range = prepaint_start.clone()..prepaint_end.clone();

                    return ((None, splices, prepaint_start..prepaint_end), element_state);
                }

                let refreshing = mem::replace(&mut window.refreshing, true);
                let prepaint_start = window.prepaint_index();
                let (element, accessed_entities) = cx.detect_accessed_entities(|cx| {
                    let mut element = render(window, cx);
                    element.layout_as_root(Size::<AvailableSpace>::from(bounds.size), window, cx);
                    element.prepaint_at(bounds.origin, window, cx);
                    element
                });

                let prepaint_end = window.prepaint_index();
                window.refreshing = refreshing;

                (
                    (
                        Some(element),
                        Vec::new(),
                        prepaint_start.clone()..prepaint_end.clone(),
                    ),
                    ViewElementState {
                        accessed_entities,
                        prepaint_range: prepaint_start..prepaint_end,
                        paint_range: PaintIndex::default()..PaintIndex::default(),
                        cache_key: ViewElementCacheKey {
                            bounds,
                            content_mask,
                            text_style,
                            element_transformation,
                        },
                    },
                )
            },
        );
        // An isolated view records where it drew, unless its whole frame was
        // reused (the reuse carried its record over).
        if let Some(render) = island
            && (element.is_some() || !window.has_island(global_id.unwrap()))
        {
            let record = window.island_here(
                entity_id,
                render.clone(),
                global_id.unwrap().clone(),
                bounds,
                range,
            );
            window.push_island(record);
        }
        ViewPrepaint { element, splices }
    })
}

/// Reuse the previous frame's prepaint `range` of a cached view, drawing the
/// isolated views inside it that were notified again in place (see
/// [`ViewElement::isolated`]).
fn reuse_prepaint_around_islands(
    range: Range<PrepaintStateIndex>,
    window: &mut Window,
    cx: &mut App,
) -> Vec<Splice> {
    let islands = window.dirty_islands_within(&range);
    if islands.is_empty() {
        window.reuse_prepaint(range);
        return Vec::new();
    }
    let mut cursor = crate::key_dispatch::ReuseCursor::default();
    let mut at = range.start.clone();
    let mut splices = Vec::with_capacity(islands.len());
    let mut segments = Vec::with_capacity(islands.len() + 1);
    for island in islands {
        let segment = at..island.prepaint_range.start.clone();
        segments.push((segment.clone(), window.prepaint_index()));
        window.reuse_prepaint_segment(&mut cursor, segment);
        let old_node = window.island_parent_node(&island);
        window.reuse_close_until(&mut cursor, old_node);
        let node = old_node.map(|n| cursor.refresh_node_id(n));
        let element = window.with_island_state(&island, |window| {
            let mut none = None;
            prepaint_view(
                island.view_id,
                Some(&island.global_id),
                island.bounds,
                &mut none,
                Some(&island.render),
                window,
                cx,
                &mut |window, cx| (island.render)(window, cx),
            )
            .element
        });
        at = island.prepaint_range.end.clone();
        splices.push(Splice {
            island,
            node,
            element,
        });
    }
    let segment = at..range.end.clone();
    segments.push((segment.clone(), window.prepaint_index()));
    window.reuse_prepaint_segment(&mut cursor, segment);
    let spliced: Vec<crate::window::Island> = splices.iter().map(|s| s.island.clone()).collect();
    window.carry_islands_around(&range, &spliced, &segments);
    window.finish_reuse_prepaint(cursor);
    splices
}

#[inline(never)]
fn prepaint_component(
    name: &'static str,
    element: &mut Option<AnyElement>,
    window: &mut Window,
    cx: &mut App,
) -> Option<AnyElement> {
    window.with_id(ElementId::from(name), |window| {
        element.as_mut().unwrap().prepaint(window, cx);
    });
    Some(element.take().unwrap())
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn paint_view(
    entity_id: EntityId,
    cached: bool,
    isolated: bool,
    global_id: Option<&GlobalElementId>,
    prepaint: &mut ViewPrepaint,
    window: &mut Window,
    cx: &mut App,
) {
    window.with_rendered_view(entity_id, |window| {
        let caching_disabled = window.is_inspector_picking(cx);
        let paint_start = window.paint_index();
        if cached && !caching_disabled {
            window.with_element_state::<ViewElementState, _>(
                global_id.unwrap(),
                |element_state, window| {
                    let mut element_state = element_state.unwrap();

                    let paint_start = window.paint_index();

                    if let Some(element) = prepaint.element.as_mut() {
                        let refreshing = mem::replace(&mut window.refreshing, true);
                        element.paint(window, cx);
                        window.refreshing = refreshing;
                    } else {
                        // Reused, with the isolated views drawn again in their places.
                        let mut at = element_state.paint_range.start.clone();
                        let mut segments = Vec::with_capacity(prepaint.splices.len() + 1);
                        for splice in prepaint.splices.iter_mut() {
                            let segment = at..splice.island.paint_range.start.clone();
                            segments.push((segment.clone(), window.paint_index()));
                            window.reuse_paint(segment);
                            if let Some(node) = splice.node {
                                window.next_frame.dispatch_tree.set_active_node(node);
                            }
                            let island = &splice.island;
                            let element = &mut splice.element;
                            window.with_island_state(island, |window| {
                                let mut drawn = ViewPrepaint {
                                    element: element.take(),
                                    splices: Vec::new(),
                                };
                                paint_view(
                                    island.view_id,
                                    true,
                                    true,
                                    Some(&island.global_id),
                                    &mut drawn,
                                    window,
                                    cx,
                                );
                            });
                            at = splice.island.paint_range.end.clone();
                        }
                        let segment = at..element_state.paint_range.end.clone();
                        segments.push((segment.clone(), window.paint_index()));
                        window.reuse_paint(segment);
                        let spliced: Vec<crate::window::Island> =
                            prepaint.splices.iter().map(|s| s.island.clone()).collect();
                        window.carry_island_paints_around(&spliced, &segments);
                    }

                    let paint_end = window.paint_index();
                    element_state.paint_range = paint_start..paint_end;

                    ((), element_state)
                },
            )
        } else {
            prepaint.element.as_mut().unwrap().paint(window, cx);
        }
        if isolated {
            let range = paint_start..window.paint_index();
            let opacity = window.element_opacity;
            if let Some(global_id) = global_id {
                window.note_island_paint(global_id, range, opacity);
            }
        }
    });
}

#[inline(never)]
fn paint_component(
    name: &'static str,
    element: &mut Option<AnyElement>,
    window: &mut Window,
    cx: &mut App,
) {
    window.with_id(ElementId::Name(name.into()), |window| {
        element.as_mut().unwrap().paint(window, cx);
    });
}

#[cfg(test)]
mod isolation_tests {
    use crate::{
        Context, Entity, Hsla, Modifiers, StyleRefinement, TestAppContext, VisualTestContext,
        Window, div, point, prelude::*, px, rgb, size,
    };
    use std::{cell::Cell, rc::Rc};

    /// A canvas that redraws often: an isolated view.
    struct Canvas {
        renders: Rc<Cell<usize>>,
        color: Hsla,
    }

    impl Render for Canvas {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            self.renders.set(self.renders.get() + 1);
            div()
                .id("canvas")
                .debug_selector(|| "CANVAS".into())
                .size_full()
                .bg(self.color)
                .child(
                    div()
                        .id("canvas-dot")
                        .debug_selector(|| "DOT".into())
                        .size(px(10.)),
                )
        }
    }

    /// Chrome around it: panels, a button after the canvas in paint order,
    /// and an overlay above it.
    struct Chrome {
        renders: Rc<Cell<usize>>,
        clicks: Rc<Cell<usize>>,
        canvas: Entity<Canvas>,
    }

    impl Render for Chrome {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            self.renders.set(self.renders.get() + 1);
            let clicks = self.clicks.clone();
            div()
                .key_context("Chrome")
                .size_full()
                .flex()
                .child(
                    div()
                        .w(px(100.))
                        .h_full()
                        .bg(rgb(0x202020))
                        .debug_selector(|| "LEFT".into()),
                )
                .child(
                    div()
                        .relative()
                        .w(px(300.))
                        .h(px(300.))
                        .child(
                            self.canvas
                                .clone()
                                .isolated(StyleRefinement::default().size_full()),
                        )
                        .child(
                            div()
                                .id("overlay")
                                .absolute()
                                .top(px(10.))
                                .left(px(10.))
                                .size(px(40.))
                                .bg(rgb(0x00ff00))
                                .debug_selector(|| "OVERLAY".into())
                                .on_click(move |_, _, _| clicks.set(clicks.get() + 1)),
                        ),
                )
                .child(
                    div()
                        .id("right")
                        .w(px(80.))
                        .h_full()
                        .debug_selector(|| "RIGHT".into()),
                )
        }
    }

    fn quad_colors(cx: &mut VisualTestContext) -> Vec<Hsla> {
        cx.update(|window, _| {
            window
                .rendered_frame
                .scene
                .quads
                .iter()
                .map(|q| q.background.solid)
                .collect()
        })
    }

    fn bounds(cx: &mut VisualTestContext, selector: &str) -> Option<crate::Bounds<crate::Pixels>> {
        cx.update(|window, _| window.rendered_frame.debug_bounds.get(selector).copied())
    }

    #[gpui::test]
    fn an_isolated_view_redraws_alone_inside_its_reused_ancestors(cx: &mut TestAppContext) {
        let (chrome_renders, canvas_renders, clicks) = (
            Rc::new(Cell::new(0)),
            Rc::new(Cell::new(0)),
            Rc::new(Cell::new(0)),
        );
        let (chrome, cx) = cx.add_window_view(|_, cx| {
            let canvas = cx.new(|_| Canvas {
                renders: canvas_renders.clone(),
                color: rgb(0x0000ff).into(),
            });
            Chrome {
                renders: chrome_renders.clone(),
                clicks: clicks.clone(),
                canvas,
            }
        });
        cx.simulate_resize(size(px(600.), px(400.)));
        cx.update(|window, _| window.set_root_cached(true));
        cx.run_until_parked();
        let (r0, c0) = (chrome_renders.get(), canvas_renders.get());
        let before = (
            bounds(cx, "LEFT"),
            bounds(cx, "CANVAS"),
            bounds(cx, "OVERLAY"),
            bounds(cx, "RIGHT"),
        );
        assert!(before.1.is_some() && before.2.is_some());
        let canvas = chrome.update(cx, |c, _| c.canvas.clone());
        let canvas_id = canvas.entity_id();

        for (k, color) in [0xff0000u32, 0x00ffff, 0xffff00].into_iter().enumerate() {
            canvas.update(cx, |c, cx| {
                c.color = rgb(color).into();
                cx.notify();
            });
            cx.run_until_parked();
            assert_eq!(chrome_renders.get(), r0, "the chrome is reused");
            assert_eq!(canvas_renders.get(), c0 + k + 1, "the canvas draws again");
            let colors = quad_colors(cx);
            let new: Hsla = rgb(color).into();
            let green: Hsla = rgb(0x00ff00).into();
            let at = |c: Hsla| colors.iter().position(|x| *x == c);
            assert!(at(new).is_some(), "the canvas's new color is painted");
            assert!(at(new) < at(green), "under the overlay, as before");
            assert_eq!(colors.iter().filter(|c| **c == new).count(), 1, "once");
            assert_eq!(
                (
                    bounds(cx, "LEFT"),
                    bounds(cx, "CANVAS"),
                    bounds(cx, "OVERLAY"),
                    bounds(cx, "RIGHT")
                ),
                before
            );
            assert!(bounds(cx, "DOT").is_some());
            // The dispatch tree keeps the canvas under the chrome.
            let path: Vec<_> = cx.update(|window, _| {
                window
                    .rendered_frame
                    .dispatch_tree
                    .view_path_reversed(canvas_id)
                    .collect()
            });
            assert_eq!(path.len(), 2, "canvas, then chrome: {path:?}");
        }
        // Listeners after the canvas in the frame still fire (reused around it).
        let overlay = before.2.unwrap();
        cx.simulate_click(overlay.center(), Modifiers::default());
        assert_eq!(clicks.get(), 1);

        // The chrome notified: it renders (and the canvas with it, as any
        // cached view inside a view that renders).
        let r1 = chrome_renders.get();
        chrome.update(cx, |_, cx| cx.notify());
        cx.run_until_parked();
        assert_eq!(chrome_renders.get(), r1 + 1);
        // And the canvas again alone, from records made in that frame.
        let c1 = canvas_renders.get();
        canvas.update(cx, |c, cx| {
            c.color = rgb(0x123456).into();
            cx.notify();
        });
        cx.run_until_parked();
        assert_eq!(
            (chrome_renders.get(), canvas_renders.get()),
            (r1 + 1, c1 + 1)
        );
        assert!(quad_colors(cx).contains(&rgb(0x123456).into()));
        cx.simulate_click(overlay.center(), Modifiers::default());
        assert_eq!(clicks.get(), 2);
    }

    /// An isolated view with an isolated view inside it (a board canvas with a pulse layer):
    /// the inner one redraws alone, and then the outer one still redraws alone: its record
    /// survives a frame that reused it around the inner one.
    struct Outer {
        renders: Rc<Cell<usize>>,
        clicks: Rc<Cell<usize>>,
        color: Hsla,
        inner: Entity<Canvas>,
    }

    impl Render for Outer {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            self.renders.set(self.renders.get() + 1);
            let clicks = self.clicks.clone();
            div()
                .relative()
                .size_full()
                .child(div().size(px(20.)).bg(self.color).debug_selector(|| "OUTER".into()))
                .child(
                    div()
                        .size(px(100.))
                        .child(self.inner.clone().isolated(StyleRefinement::default().size_full())),
                )
                .child(
                    div()
                        .id("after-inner")
                        .absolute()
                        .top(px(150.))
                        .left(px(0.))
                        .size(px(30.))
                        .bg(rgb(0x00ff00))
                        .debug_selector(|| "AFTER".into())
                        .on_click(move |_, _, _| clicks.set(clicks.get() + 1)),
                )
        }
    }

    struct Shell {
        renders: Rc<Cell<usize>>,
        outer: Entity<Outer>,
    }

    impl Render for Shell {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            self.renders.set(self.renders.get() + 1);
            div()
                .size_full()
                .flex()
                .child(div().w(px(50.)).h_full().bg(rgb(0x202020)))
                .child(div().size(px(300.)).child(self.outer.clone().isolated(StyleRefinement::default().size_full())))
        }
    }

    #[gpui::test]
    fn an_isolated_view_around_one_drawn_alone_still_redraws_alone(cx: &mut TestAppContext) {
        let (shell_renders, outer_renders, inner_renders, clicks) =
            (Rc::new(Cell::new(0)), Rc::new(Cell::new(0)), Rc::new(Cell::new(0)), Rc::new(Cell::new(0)));
        let (shell, cx) = cx.add_window_view(|_, cx| {
            let inner = cx.new(|_| Canvas { renders: inner_renders.clone(), color: rgb(0x0000ff).into() });
            let outer = cx.new(|_| Outer {
                renders: outer_renders.clone(),
                clicks: clicks.clone(),
                color: rgb(0x880000).into(),
                inner,
            });
            Shell { renders: shell_renders.clone(), outer }
        });
        cx.simulate_resize(size(px(600.), px(400.)));
        cx.update(|window, _| window.set_root_cached(true));
        cx.run_until_parked();
        let outer = shell.update(cx, |s, _| s.outer.clone());
        let inner = outer.update(cx, |o, _| o.inner.clone());
        let s0 = shell_renders.get();
        let before = (bounds(cx, "OUTER"), bounds(cx, "CANVAS"), bounds(cx, "AFTER"));
        for (k, (inner_color, outer_color)) in [(0xff0000u32, 0x111111u32), (0x00ffff, 0x222222), (0xffff00, 0x333333)].into_iter().enumerate() {
            // The inner view alone: the outer one is reused around it.
            let (o0, i0) = (outer_renders.get(), inner_renders.get());
            inner.update(cx, |c, cx| {
                c.color = rgb(inner_color).into();
                cx.notify();
            });
            cx.run_until_parked();
            assert_eq!((shell_renders.get(), outer_renders.get(), inner_renders.get()), (s0, o0, i0 + 1), "round {k}: the inner view alone");
            // Then the outer one: it redraws alone (with the inner one inside it), not the shell.
            outer.update(cx, |o, cx| {
                o.color = rgb(outer_color).into();
                cx.notify();
            });
            cx.run_until_parked();
            assert_eq!(shell_renders.get(), s0, "round {k}: the shell is reused");
            assert_eq!(outer_renders.get(), o0 + 1, "round {k}: the outer view draws again");
            let colors = quad_colors(cx);
            let (new_outer, new_inner, green): (Hsla, Hsla, Hsla) = (rgb(outer_color).into(), rgb(inner_color).into(), rgb(0x00ff00).into());
            let at = |c: Hsla| colors.iter().position(|x| *x == c);
            assert_eq!(colors.iter().filter(|c| **c == new_outer).count(), 1, "round {k}: the outer color, once");
            assert_eq!(colors.iter().filter(|c| **c == new_inner).count(), 1, "round {k}: the inner color, once");
            assert!(at(new_outer) < at(new_inner) && at(new_inner) < at(green), "round {k}: paint order kept");
            assert_eq!((bounds(cx, "OUTER"), bounds(cx, "CANVAS"), bounds(cx, "AFTER")), before, "round {k}: same places");
        }
        // The listener after the inner view, inside the outer one, still fires.
        cx.simulate_click(before.2.unwrap().center(), Modifiers::default());
        assert_eq!(clicks.get(), 1);
    }

    /// A pad of buttons: one with an active style and a click listener, one with neither.
    struct Pad {
        renders: Rc<Cell<usize>>,
        clicks: Rc<Cell<usize>>,
    }

    impl Render for Pad {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            self.renders.set(self.renders.get() + 1);
            let clicks = self.clicks.clone();
            div()
                .id("pad")
                .size_full()
                .flex()
                .child(
                    div()
                        .id("button")
                        .debug_selector(|| "BUTTON".into())
                        .size(px(40.))
                        .bg(rgb(0x404040))
                        .active(|s| s.bg(rgb(0xff0000)))
                        .on_click(move |_, _, _| clicks.set(clicks.get() + 1)),
                )
                .child(
                    div()
                        .id("plain")
                        .debug_selector(|| "PLAIN".into())
                        .size(px(40.))
                        .bg(rgb(0x00ff00)),
                )
        }
    }

    /// Chrome around the pad, with a plain element of its own.
    struct PadShell {
        renders: Rc<Cell<usize>>,
        pad: Entity<Pad>,
    }

    impl Render for PadShell {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            self.renders.set(self.renders.get() + 1);
            div()
                .id("shell")
                .size_full()
                .flex()
                .child(
                    div()
                        .id("shell-plain")
                        .debug_selector(|| "SHELL_PLAIN".into())
                        .size(px(40.))
                        .bg(rgb(0x202020)),
                )
                .child(
                    div()
                        .size(px(200.))
                        .child(self.pad.clone().isolated(StyleRefinement::default().size_full())),
                )
        }
    }

    #[gpui::test]
    fn a_press_redraws_only_the_view_it_lands_in(cx: &mut TestAppContext) {
        use crate::MouseButton;
        let (shell_renders, pad_renders, clicks) =
            (Rc::new(Cell::new(0)), Rc::new(Cell::new(0)), Rc::new(Cell::new(0)));
        let (_shell, cx) = cx.add_window_view(|_, cx| {
            let pad = cx.new(|_| Pad { renders: pad_renders.clone(), clicks: clicks.clone() });
            PadShell { renders: shell_renders.clone(), pad }
        });
        cx.simulate_resize(size(px(600.), px(400.)));
        cx.update(|window, _| window.set_root_cached(true));
        cx.run_until_parked();
        let button = bounds(cx, "BUTTON").unwrap().center();
        let (red, gray): (Hsla, Hsla) = (rgb(0xff0000).into(), rgb(0x404040).into());
        let m = Modifiers::default();
        for k in 1..=3 {
            let (s0, p0) = (shell_renders.get(), pad_renders.get());
            cx.simulate_mouse_move(button, None, m);
            cx.simulate_mouse_down(button, MouseButton::Left, m);
            cx.run_until_parked();
            assert_eq!(shell_renders.get(), s0, "round {k}: a press keeps the chrome");
            assert!(pad_renders.get() > p0, "round {k}: the pad draws its active style");
            let colors = quad_colors(cx);
            assert!(colors.contains(&red) && !colors.contains(&gray), "round {k}: pressed: {colors:?}");
            let p1 = pad_renders.get();
            cx.simulate_mouse_up(button, MouseButton::Left, m);
            cx.run_until_parked();
            assert_eq!(shell_renders.get(), s0, "round {k}: a release keeps the chrome");
            assert!(pad_renders.get() > p1, "round {k}: the pad draws without its active style");
            let colors = quad_colors(cx);
            assert!(colors.contains(&gray) && !colors.contains(&red), "round {k}: released: {colors:?}");
            assert_eq!(clicks.get(), k, "round {k}: the click fires");
        }
        // A press and release on elements without an active style or click listener draw nothing.
        for name in ["PLAIN", "SHELL_PLAIN"] {
            let at = bounds(cx, name).unwrap().center();
            let (s0, p0) = (shell_renders.get(), pad_renders.get());
            cx.simulate_mouse_move(at, None, m);
            cx.simulate_click(at, m);
            cx.run_until_parked();
            assert_eq!((shell_renders.get(), pad_renders.get()), (s0, p0), "{name}: nothing draws");
        }
    }

    use crate as gpui;
    gpui::actions!(isolation_test, [Ping]);

    /// A canvas with a focused element, a varying number of children (so its
    /// dispatch nodes change count between frames), under chrome that handles
    /// an action and opens a deferred popover after it.
    struct Focused {
        focus: crate::FocusHandle,
        children: usize,
    }

    impl Render for Focused {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            div()
                .id("focused-canvas")
                .key_context("Canvas")
                .track_focus(&self.focus)
                .size_full()
                .children(
                    (0..self.children)
                        .map(|i| div().id(("kid", i)).key_context("Kid").size(px(4.))),
                )
        }
    }

    struct ActionChrome {
        canvas: Entity<Focused>,
        pings: Rc<Cell<usize>>,
        renders: Rc<Cell<usize>>,
    }

    impl Render for ActionChrome {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            let pings = self.pings.clone();
            self.renders.set(self.renders.get() + 1);
            div()
                .key_context("Chrome")
                .on_action(move |_: &Ping, _, _| pings.set(pings.get() + 1))
                .size_full()
                .child(
                    div().size(px(200.)).child(
                        self.canvas
                            .clone()
                            .isolated(StyleRefinement::default().size_full()),
                    ),
                )
                .child(crate::deferred(
                    crate::anchored().position(point(px(300.), px(10.))).child(
                        div()
                            .key_context("Popover")
                            .debug_selector(|| "POPOVER".into())
                            .size(px(30.)),
                    ),
                ))
        }
    }

    #[gpui::test]
    fn actions_focus_and_deferred_draws_survive_isolated_redraws(cx: &mut TestAppContext) {
        let (pings, renders) = (Rc::new(Cell::new(0)), Rc::new(Cell::new(0)));
        let (chrome, cx) = cx.add_window_view(|_, cx| {
            let canvas = cx.new(|cx| Focused {
                focus: cx.focus_handle(),
                children: 1,
            });
            ActionChrome {
                canvas,
                pings: pings.clone(),
                renders: renders.clone(),
            }
        });
        cx.update(|window, _| window.set_root_cached(true));
        let canvas = chrome.update(cx, |c, _| c.canvas.clone());
        cx.update(|window, cx| window.focus(&canvas.read(cx).focus.clone(), cx));
        cx.run_until_parked();
        let r0 = renders.get();
        for n in [5usize, 0, 3, 9, 2] {
            canvas.update(cx, |c, cx| {
                c.children = n;
                cx.notify();
            });
            cx.run_until_parked();
            assert_eq!(renders.get(), r0, "the chrome is reused");
            let focused = cx.update(|window, cx| canvas.read(cx).focus.is_focused(window));
            assert!(focused, "focus stays in the redrawn canvas");
            let before = pings.get();
            cx.dispatch_action(Ping);
            assert_eq!(
                pings.get(),
                before + 1,
                "the chrome's action handler is still on the focus path"
            );
            let popover =
                cx.update(|window, _| window.rendered_frame.debug_bounds.get("POPOVER").copied());
            assert!(
                popover.is_some(),
                "the deferred popover after the canvas is drawn"
            );
        }
    }

    /// A view drawn at two places (one isolated, one not) is not isolated:
    /// its notify renders the ancestors, so both mounts draw it again.
    #[gpui::test]
    fn a_view_mounted_twice_is_not_isolated(cx: &mut TestAppContext) {
        struct Twice {
            renders: Rc<Cell<usize>>,
            canvas: Entity<Canvas>,
        }
        impl Render for Twice {
            fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
                self.renders.set(self.renders.get() + 1);
                div()
                    .size_full()
                    .child(div().size(px(50.)).child(self.canvas.clone().isolated(StyleRefinement::default().size_full())))
                    .child(div().size(px(50.)).child(self.canvas.clone().cached(StyleRefinement::default().size_full())))
            }
        }
        let renders = Rc::new(Cell::new(0));
        let (root, cx) = cx.add_window_view(|_, cx| Twice {
            renders: renders.clone(),
            canvas: cx.new(|_| Canvas { renders: Rc::default(), color: rgb(0x0000ff).into() }),
        });
        cx.update(|window, _| window.set_root_cached(true));
        cx.run_until_parked();
        let r0 = renders.get();
        let canvas = root.update(cx, |r, _| r.canvas.clone());
        canvas.update(cx, |c, cx| {
            c.color = rgb(0xff0000).into();
            cx.notify();
        });
        cx.run_until_parked();
        assert_eq!(renders.get(), r0 + 1, "the root renders");
        let red: Hsla = rgb(0xff0000).into();
        assert_eq!(quad_colors(cx).iter().filter(|c| **c == red).count(), 2, "both mounts show the change");
    }

    /// A text field in the chrome: its input handler is painted each frame.
    struct Field {
        focus: crate::FocusHandle,
        text: String,
    }

    impl Render for Field {
        fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
            let (view, focus) = (cx.entity(), self.focus.clone());
            div().track_focus(&self.focus).size(px(80.)).child(
                crate::canvas(
                    |_, _, _| {},
                    move |bounds, _, window, cx| window.handle_input(&focus, crate::ElementInputHandler::new(bounds, view), cx),
                )
                .size_full(),
            )
        }
    }

    impl crate::EntityInputHandler for Field {
        fn text_for_range(&mut self, _: std::ops::Range<usize>, _: &mut Option<std::ops::Range<usize>>, _: &mut Window, _: &mut Context<Self>) -> Option<String> {
            Some(self.text.clone())
        }
        fn selected_text_range(&mut self, _: bool, _: &mut Window, _: &mut Context<Self>) -> Option<crate::UTF16Selection> {
            let n = self.text.encode_utf16().count();
            Some(crate::UTF16Selection { range: n..n, reversed: false })
        }
        fn marked_text_range(&self, _: &mut Window, _: &mut Context<Self>) -> Option<std::ops::Range<usize>> {
            None
        }
        fn unmark_text(&mut self, _: &mut Window, _: &mut Context<Self>) {}
        fn replace_text_in_range(&mut self, _: Option<std::ops::Range<usize>>, text: &str, _: &mut Window, _: &mut Context<Self>) {
            self.text.push_str(text);
        }
        fn replace_and_mark_text_in_range(&mut self, _: Option<std::ops::Range<usize>>, _: &str, _: Option<std::ops::Range<usize>>, _: &mut Window, _: &mut Context<Self>) {}
        fn bounds_for_range(&mut self, _: std::ops::Range<usize>, bounds: crate::Bounds<crate::Pixels>, _: &mut Window, _: &mut Context<Self>) -> Option<crate::Bounds<crate::Pixels>> {
            Some(bounds)
        }
        fn character_index_for_point(&mut self, _: crate::Point<crate::Pixels>, _: &mut Window, _: &mut Context<Self>) -> Option<usize> {
            None
        }
    }

    struct Tip;

    impl Render for Tip {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            div().debug_selector(|| "TIP".into()).size(px(20.))
        }
    }

    /// A canvas with its own deferred menu, redrawn alone.
    struct MenuCanvas {
        renders: Rc<Cell<usize>>,
    }

    impl Render for MenuCanvas {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            self.renders.set(self.renders.get() + 1);
            div().id("menu-canvas").debug_selector(|| "CANVAS_AREA".into()).size_full().cursor(crate::CursorStyle::Crosshair).child(crate::deferred(
                crate::anchored()
                    .position(point(px(5.), px(5.)))
                    .child(div().debug_selector(|| "CANVAS_MENU".into()).size(px(15.))),
            ))
        }
    }

    struct RichChrome {
        renders: Rc<Cell<usize>>,
        field: Entity<Field>,
        canvas: Entity<MenuCanvas>,
    }

    impl Render for RichChrome {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            self.renders.set(self.renders.get() + 1);
            div()
                .size_full()
                .flex()
                .child(self.field.clone())
                .child(div().id("link").size(px(40.)).cursor_pointer().debug_selector(|| "LINK".into()))
                .child(
                    div()
                        .id("tip-target")
                        .size(px(40.))
                        .debug_selector(|| "TIP_TARGET".into())
                        .tooltip(|_, cx| cx.new(|_| Tip).into()),
                )
                .child(div().size(px(200.)).child(self.canvas.clone().isolated(StyleRefinement::default().size_full())))
        }
    }

    /// While the canvas redraws alone, the chrome's text field keeps its
    /// input handler, its hitboxes their cursors, a shown tooltip stays, and
    /// the canvas's own deferred menu draws.
    #[gpui::test]
    fn input_cursors_tooltips_and_deferred_children_survive_isolated_redraws(cx: &mut TestAppContext) {
        let (chrome_renders, canvas_renders) = (Rc::new(Cell::new(0)), Rc::new(Cell::new(0)));
        let (chrome, cx) = cx.add_window_view(|_, cx| RichChrome {
            renders: chrome_renders.clone(),
            field: cx.new(|cx| Field { focus: cx.focus_handle(), text: String::new() }),
            canvas: cx.new(|_| MenuCanvas { renders: canvas_renders.clone() }),
        });
        cx.simulate_resize(size(px(600.), px(400.)));
        cx.update(|window, _| window.set_root_cached(true));
        let (field, canvas) = chrome.update(cx, |c, _| (c.field.clone(), c.canvas.clone()));
        cx.update(|window, cx| window.focus(&field.read(cx).focus.clone(), cx));
        cx.run_until_parked();
        // Show the tooltip.
        let tip_target = bounds(cx, "TIP_TARGET").unwrap();
        cx.simulate_mouse_move(tip_target.center(), None, Modifiers::default());
        cx.executor().advance_clock(std::time::Duration::from_millis(800));
        cx.run_until_parked();
        assert!(bounds(cx, "TIP").is_some(), "the tooltip shows");
        let r0 = chrome_renders.get();
        for k in 0..3 {
            let c0 = canvas_renders.get();
            canvas.update(cx, |_, cx| cx.notify());
            cx.run_until_parked();
            assert_eq!(canvas_renders.get(), c0 + 1, "round {k}: the canvas redraws");
            assert_eq!(chrome_renders.get(), r0, "round {k}: the chrome is reused");
            assert!(bounds(cx, "TIP").is_some(), "round {k}: the tooltip stays");
            assert!(bounds(cx, "CANVAS_MENU").is_some(), "round {k}: the canvas's deferred menu draws");
        }
        // The field's input handler (painted in the reused chrome frame).
        cx.simulate_input("hi");
        assert_eq!(field.read_with(cx, |f, _| f.text.clone()), "hi");
        // Cursors: the chrome's link (reused hitbox) and the redrawn canvas.
        let link = bounds(cx, "LINK").unwrap();
        cx.simulate_mouse_move(link.center(), None, Modifiers::default());
        assert_eq!(cx.update(|window, _| window.rendered_frame.cursor_style(window)), Some(crate::CursorStyle::PointingHand));
        let area = bounds(cx, "CANVAS_AREA").unwrap();
        cx.simulate_mouse_move(point(area.center().x, area.origin.y + area.size.height - px(10.)), None, Modifiers::default());
        assert_eq!(cx.update(|window, _| window.rendered_frame.cursor_style(window)), Some(crate::CursorStyle::Crosshair));
        canvas.update(cx, |_, cx| cx.notify());
        cx.run_until_parked();
        assert_eq!(cx.update(|window, _| window.rendered_frame.cursor_style(window)), Some(crate::CursorStyle::Crosshair));
    }

    /// Without isolation (and by default), a notify re-renders the ancestors.
    #[gpui::test]
    fn a_cached_view_s_notify_still_renders_its_ancestors(cx: &mut TestAppContext) {
        struct Root {
            renders: Rc<Cell<usize>>,
            child: Entity<Canvas>,
        }
        impl Render for Root {
            fn render(
                &mut self,
                _window: &mut Window,
                _cx: &mut Context<Self>,
            ) -> impl IntoElement {
                self.renders.set(self.renders.get() + 1);
                div().size_full().child(
                    self.child
                        .clone()
                        .cached(StyleRefinement::default().size_full()),
                )
            }
        }
        let renders = Rc::new(Cell::new(0));
        let (root, cx) = cx.add_window_view(|_, cx| Root {
            renders: renders.clone(),
            child: cx.new(|_| Canvas {
                renders: Rc::default(),
                color: rgb(0).into(),
            }),
        });
        cx.update(|window, _| window.set_root_cached(true));
        cx.run_until_parked();
        let r0 = renders.get();
        let child = root.update(cx, |r, _| r.child.clone());
        child.update(cx, |_, cx| cx.notify());
        cx.run_until_parked();
        assert_eq!(renders.get(), r0 + 1);
        let _ = point(px(0.), px(0.));
    }
}
