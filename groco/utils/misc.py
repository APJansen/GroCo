def backup_and_restore(attribute_names):
    """
    Decorator that backs up given attributes, runs the function, and then restores the backed up values.
    """

    def decorator(function):
        def wrapper(self, inputs):
            backup = {name: getattr(self, name) for name in attribute_names}
            result = function(self, inputs)
            for name in attribute_names:
                setattr(self, name, backup[name])
            return result

        return wrapper

    return decorator
