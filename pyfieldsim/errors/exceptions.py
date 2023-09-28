class WrongShapeError(Exception):
    def __str__(self):
        return "Shape of sky field is incorrect. " \
               "Must be a `tuple` of length 2."


class WrongDataFileError(Exception):
    def __init__(self, msg=None):
        if msg is None:
            self.msg = "Wrong data file."
        else:
            self.msg = msg

    def __str__(self):
        return self.msg
