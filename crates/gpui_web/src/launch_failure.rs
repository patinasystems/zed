use std::cell::RefCell;

/// The application's handler for a launch that cannot finish, for example
/// when no browser graphics backend starts. It runs at most once.
pub(crate) struct LaunchFailure<E> {
    handler: RefCell<Option<Box<dyn FnOnce(E)>>>,
}

impl<E> Default for LaunchFailure<E> {
    fn default() -> Self {
        Self {
            handler: RefCell::new(None),
        }
    }
}

impl<E> LaunchFailure<E> {
    /// Replaces the handler.
    pub(crate) fn set(&self, handler: Box<dyn FnOnce(E)>) {
        *self.handler.borrow_mut() = Some(handler);
    }

    /// Gives `error` to the handler and returns `None`, or returns `error`
    /// when no handler is set (or it already ran).
    pub(crate) fn report(&self, error: E) -> Option<E> {
        let handler = self.handler.borrow_mut().take();
        match handler {
            Some(handler) => {
                handler(error);
                None
            }
            None => Some(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::LaunchFailure;
    use std::{cell::RefCell, rc::Rc};

    #[test]
    fn report_without_handler_returns_the_error() {
        let failure = LaunchFailure::<String>::default();
        assert_eq!(
            failure.report("no backend".into()),
            Some("no backend".into())
        );
    }

    #[test]
    fn report_gives_the_error_to_the_handler_once() {
        let failure = LaunchFailure::<String>::default();
        let heard = Rc::new(RefCell::new(Vec::new()));
        let sink = heard.clone();
        failure.set(Box::new(move |error| sink.borrow_mut().push(error)));
        assert_eq!(failure.report("no backend".into()), None);
        assert_eq!(*heard.borrow(), vec!["no backend".to_string()]);
        assert_eq!(failure.report("again".into()), Some("again".into()));
        assert_eq!(heard.borrow().len(), 1);
    }

    #[test]
    fn set_replaces_the_handler() {
        let failure = LaunchFailure::<u32>::default();
        let heard = Rc::new(RefCell::new(Vec::new()));
        let first = heard.clone();
        failure.set(Box::new(move |n| first.borrow_mut().push(("first", n))));
        let second = heard.clone();
        failure.set(Box::new(move |n| second.borrow_mut().push(("second", n))));
        assert_eq!(failure.report(7), None);
        assert_eq!(*heard.borrow(), vec![("second", 7)]);
    }
}
