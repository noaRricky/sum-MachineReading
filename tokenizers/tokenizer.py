"""Base tokenizer/tokens classes and utilities
"""


class Tokenzier(object):
    """Base tokenizer class
    """

    def tokenizer(self, text):
        raise NotImplementedError

    def shutdown(self):
        pass

    def __del__(self):
        self.shutdown()
