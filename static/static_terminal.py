import subprocess
import sys
from . import colorful as c


def visible_len(s):
    l = 0
    special = False
    for c in s:
        if c == "\033":
            special = True
        if special:
            if c == "m":
                special = False
        else:
            l += 1
    return l


# def add_wrap(self, name:str, x1: int, y1: int,
#                         x2: int, y2: int, has_border:bool,
#                         border_color_code: str = None, custom_trigger: str = "e"):
#     return f"/{custom_trigger}_add({name};;{x1};;{y1};;{x2};;{y2};;{int(has_border)};;{border_color_code})_{custom_trigger}/"
#
#
# def updc_wrap(self, name:str, line_i:int, content:str, custom_col:int=0, custom_trigger: str = "e"):
#     return f"/{custom_trigger}_updc({name};;{line_i};;{content};;{custom_col})_{custom_trigger}/"
#
#
# def updb_wrap(self, name:str, new_border:str, custom_trigger: str = "e"):
#     return f"/{custom_trigger}_updb({name};;{new_border})_{custom_trigger}/"
#
#
# def roll_wrap(self, name:str, content:str, custom_trigger: str = "e"):
#     return f"/{custom_trigger}_roll({name};;{content})_{custom_trigger}/"


# class _ScriptReceiver:
#     """
#     A reading&printing object, used to capture custom script's output into streamline, catching StaticTerminal commands. \n
#     You should not create this object in your script. Use 'StaticTerminal' instead.
#     """
#     process = None
#
#     si = None
#     pep = None
#
#     parent = None
#
#     def __init__(self, parent, python_exec_path:str=None):
#         self.parent = parent
#         if python_exec_path:
#             self.pep = python_exec_path
#         else:
#             self.pep = sys.executable
#
#     def start(self, script_path:str, quit_on_error:bool=True, custom_trigger="exec", variables:list=[]):
#         """
#         Start your script in background, while stdout is streamlined to this script's output.\n
#         If you pass specific command in script's output (print), it will execute and won't print
#         here (this is the only way to call anything in this script from the other script). All commands start with '/exec_' and end with '_exec/' by default, but you can change it with 'custom_trigger' argument. You can stack several commands by separator '||'. \n
#         Example command pass: "/exec_roll((0;;Some text))_exec/" \n
#         Main methods' shortened versions for command passes:
#          - add_rect = add \n
#          - roll_string = roll \n
#          - update_content = updc \n
#          - update_border = updb \n
#         Keyboard interrupt will automatically terminate both scripts\n
#         :param script_path: path to your Python script
#         :param quit_on_error: program will quit() after catching script exit (Not alive).  Defaults to True
#         :param custom_trigger: change triggering string combination for executing StaticTerminal commands. Commands start with '/{custom_trigger}_' and end with '_{custom_trigger}/'. Defaults to 'exec'
#         :param variables: custom env variables
#         :return: None
#         """
#         args = [self.pep, "-u", script_path]
#         for env in variables:
#             args.append(env)
#         self.process = subprocess.Popen(
#             args,
#             #creationflags=subprocess.CREATE_NEW_CONSOLE,
#             startupinfo=self.si,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.STDOUT,
#             text=True,
#             bufsize=1,
#         )
#
#         while True:
#             for line in self.process.stdout:
#                 lraw = line.strip()
#                 lout = ""
#                 com_all = []
#                 if "_"+custom_trigger+"/" in lraw and "/"+custom_trigger+"_" in lraw:
#                     split_l = lraw.split("/"+custom_trigger+"_")
#                     lout += split_l[0]
#                     for i in split_l[1:]:
#                         i_s = i.split("_"+custom_trigger+"/")
#                         lout += i_s[1]
#                         com_all.extend(i_s[0].split("||"))
#
#                 for com in com_all:
#                     self.parent.execute(com)
#
#
#             if self.process.poll() is not None:
#                 print(c.F.color(1) + "Not alive (Terminating process)")
#                 if quit_on_error:
#                     quit()

class StaticTerminal:
    fields = {}
    full_width_m = 155
    full_height_m = 45

    full_width = 119
    full_height = 29

    maximized = False
    pep = ""
    auto_update = True

    def __init__(self, maximized:bool = False, python_exec_path:str = "", auto_update:bool = True):
        self.maximized = maximized
        if self.maximized:
            self.add_rect("0", 0,0, self.full_width_m, self.full_height_m, False)
        else:
            self.add_rect("0",0,0, self.full_width, self.full_height, False)

        if python_exec_path != "":
            self.pep = python_exec_path
        else:
            self.pep = sys.executable

        self.auto_update = auto_update

    def add_rect(self, name:str, x1: int, y1: int,
                        x2: int, y2: int, has_border:bool,
                        border_color_code: str = ""):
        width = (x2 - x1)+1
        height = (y2 - y1)+1
        self.fields[name] = {"start": (x1,y1),
                            "end": (x2,y2),
                            "border": border_color_code if has_border else None,
                            "content": [" "*width for i in range(height)] if not has_border
                                    else [" " * (width-2) for i in range(height-2)],
                           }
        if self.auto_update:
            self.press()

    def press(self):
        if self.maximized:
            full_width = self.full_width_m
            full_height = self.full_height_m
        else:
            full_width = self.full_width
            full_height = self.full_height

        full_print = ""
        for line in range(full_height+1):
            string = ""
            col = 0
            fields_bonus = [0 for i in range(len(self.fields))]
            while col <= full_width:
                final_char = ""
                for fi in range(len(self.fields)):
                    field = list(self.fields.values())[fi]
                    bonus = fields_bonus[fi]
                    cline = line - field["start"][1]
                    ccol = col - field["start"][0]
                    if (field["start"][0] <= col <= field["end"][0] and
                            field["start"][1] <= line <= field["end"][1]):

                        if field["border"]:
                            if ((col == field["start"][0] or col == field["end"][0])
                                    and (line == field["start"][1] or line == field["end"][1])):
                                final_char = field["border"] + "+" + c.F.reset()
                            elif col == field["start"][0] or col == field["end"][0]:
                                final_char = field["border"] + "|" + c.F.reset()
                            elif line == field["start"][1] or line == field["end"][1]:
                                final_char = field["border"] + "—" + c.F.reset()
                            elif (field["start"][0] < col < field["end"][0]
                                  and field["start"][1] < line < field["end"][1]):
                                cline -= 1
                                ccol -= 1

                                final_char = field["content"][cline][ccol + bonus]

                        else:
                            final_char = field["content"][cline][ccol + bonus]


                        while final_char[-1] == "\033":
                            while final_char[-1] != "m":
                                fields_bonus[fi] += 1
                                bonus += 1
                                final_char += field["content"][cline][ccol + bonus]
                            fields_bonus[fi] += 1
                            bonus += 1
                            final_char += field["content"][cline][ccol + bonus]

                col += 1
                string += final_char

            full_print += string + c.F.reset() + "\n"

        print(full_print, end="", flush=True)

    def edit(self, name:str, line_i:int, content:str, custom_col:int=0):
        con_width = visible_len(self.fields[name]["content"][0])
        new_str = " "*custom_col + content + " "*(con_width - (visible_len(content)+custom_col))
        self.fields[name]["content"][line_i] = new_str
        if self.auto_update:
            self.press()

    def edit_border(self, name:str, new_border:str):
        self.fields[name]["border"] = new_border
        if self.auto_update:
            self.press()

    def roll(self, name:str, content:str):
        self.fields[name]["content"].pop(0)
        con_width = visible_len(self.fields[name]["content"][0])
        new_str = content
        while visible_len(new_str) > con_width:
            self.fields[name]["content"].append(new_str[:con_width])
            self.fields[name]["content"].pop(0)
            new_str = new_str[con_width:]
        else:
            new_str = new_str + " " * (con_width - visible_len(new_str))
            self.fields[name]["content"].append(new_str)
        if self.auto_update:
            self.press()
        # print(len(content), visible_len(content), con_width)

    # def start(self, script_path:str, custom_trigger:str="exec", variables:list=[]):
    #     s = _ScriptReceiver(self)
    #     s.start(script_path, custom_trigger=custom_trigger, variables=variables)

    # def execute(self, command:str):
    #     cs = command.split("((")
    #     name = cs[0]
    #     params = cs[1].split("))")[0].split(";;")
    #     if name == "roll":
    #         self.roll(params[0], params[1])
    #     if name == "add":
    #         if len(params) == 7:
    #             self.add_rect(params[0], int(params[1]), int(params[2]) ,int(params[3]), int(params[4]), bool(int(params[5])), border_color_code=params[6])
    #         else:
    #             self.add_rect(params[0], int(params[1]), int(params[2]), int(params[3]), int(params[4]), bool(int(params[5])))
    #     if name == "edit":
    #         if len(params) == 4:
    #             self.edit(params[0], int(params[1]), params[2], custom_col=int(params[3]))
    #         else:
    #             self.edit(params[0], int(params[1]), params[2])
    #     if name == "edbo":
    #         self.edit_border(params[0], params[1])
    #     if name == "press":
    #         self.press()