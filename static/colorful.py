class F:
    @staticmethod
    def color(code):
        """
        Return ANSI escape code for selected color for Foreground.
        :param code:
        :return: string ANSI escape code
        """
        return f"\033[38;5;{code}m"

    @staticmethod
    def reset():
        """
        Return ANSI escape code for default color.
        :return: string ANSI escape code
        """
        return "\033[0m"

class B:
    @staticmethod
    def color(code):
        """
        Return ANSI escape code for selected color for Background.
        :param code:
        :return: string ANSI escape code
        """
        return f"\033[48;5;{code}m"

    @staticmethod
    def reset():
        """
        Return ANSI escape code for default color.
        :return: string ANSI escape code
        """
        return "\033[0m"

class S:
    @staticmethod
    def style(code):
        return f"\033[{code}m"

    @staticmethod
    def reset():
        return f"\033[22m" + f"\033[23m" + f"\033[24m" + f"\033[25m" + f"\033[27m" + f"\033[29m"