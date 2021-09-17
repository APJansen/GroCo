def backup_and_restore(attribute_names):
    def decorator(function):
        def wrapper(self, inputs):
            backup = {name: getattr(self, name) for name in attribute_names}
            result = function(self, inputs)
            for name in attribute_names:
                setattr(self, name, backup[name])
            return result
        return wrapper
    return decorator