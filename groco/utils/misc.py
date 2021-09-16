def backup_and_restore_attributes(function):
    def wrapper(self, inputs):
        backup = self.kernel, self.bias, self.filters
        result = function(self, inputs)
        self.kernel, self.bias, self.filters = backup
        return result

    return wrapper