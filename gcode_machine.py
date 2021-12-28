"""
gcode_machine - Copyright (c) 2016 Michael Franzl

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
"""

import re
import logging
import math
import numpy as np

class GcodeMachine:
    """ This implements a simple CNC state machine that can be used
    for simulation and processing of G-code.
    
    For usage, see README.md.
    
    Callbacks:
    
    on_feed_change
    : Emitted when a F keyword is parsed from the G-Code.
    : 1 argument: the feed rate in mm/min

    on_var_undefined
    : Emitted when a variable is to be substituted but no substitution value has been set previously.
    : 1 argument: the key of the undefined variable
    """
    
    def __init__(self, impos=(0,0,0), ics="G54", cs_offsets={"G54":(0,0,0)}):
        """ Initialization.
        """
        
        self.logger = logging.getLogger('gerbil')
        
        
        ## @var line
        # Holds the current Gcode line
        self.line = ""
        
        ## @var vars
        # A dict holding values for variable substitution.
        self.vars = {}
        
        ## @var callback method for parent applications
        self.callback = self._default_callback
        
        ## @var do_feed_override
        # If set to True, F commands will be replaced with the feed
        # speed set in `request_feed`. If set to False, no feed speed
        # processing will be done.
        self.do_feed_override = False
        
        ## @var do_fractionize_lines
        # If set to True, linear movements over the threshold of
        # `fract_linear_threshold` will be broken down into small line
        # segments of length `fract_linear_segment_len`. If set to False
        # no processing is done on lines.
        self.do_fractionize_lines = False
        
        ## @var do_fractionize_arcs
        # If set to True, arcs will be broken up into small line segments.
        # If set to False, no processing is done on arcs.
        self.do_fractionize_arcs = False
        
        ## @var fract_linear_threshold
        # The threshold for the fractionization of lines.
        self.fract_linear_threshold = 1
        
        ## @var fract_linear_segment_len
        # The length of the segments of fractionized lines.
        self.fract_linear_segment_len = 0.5
        
        ## @var spindle_factor
        # Scale S by this value
        self.spindle_factor = 1
        
        ## @var request_feed
        # If `do_feed_override` is True, a F command will be inserted
        # to the currently active command with this value.
        self.request_feed = None

        ## @var current_feed
        # The current feed rate of the machine
        self.current_feed = None
        
        ## @var contains_feed
        # True if the current line contains a F command
        self.contains_feed = False
        
        ## @var current_distance_mode
        # Contains the current distance mode of the machine as string
        self.current_distance_mode = "G90"
        
        ## @var current_motion_mode
        # Contains the current motion mode of the machine as integer (0 to 3)
        self.current_motion_mode = 0
        
        ## @var current_plane_mode
        # Contains the current plane mode of the machine as string
        self.current_plane_mode = "G17"
        
       
        ## @var cs_offsets
        # Coordinate system offsets. A Python dict with
        # 3-tuples as offsets.
        self.cs_offsets = cs_offsets
        
        ## @var cs
        # Current coordinate system as string.
        self.cs = ics
        
        ## @var pos_m
        # Contains the current machine position before execution
        # of the currently set line (the target of the last command)
        self.pos_m = list(impos) # initial machine position
        
        ## @var pos_w
        # Contains the current working position before execution
        # of the currently set line (the target of the last command)
        self.pos_w = list(np.subtract(self.pos_m, self.cs_offsets[self.cs]))
       
        
        ## @var target_m
        # Contains the position target of the currently set command in
        # the machine coordinate system.
        self.target_m = [None, None, None]
        
        ## @var target_m
        # Contains the position target of the currently set command in
        # the currently seleted coordinate system.
        self.target_w = [None, None, None]
        
        ## @var offset
        # Contains the offset of the arc center from current position
        self.offset = [0, 0, 0] 
        
        ## @var radius
        # Contains the radius of the current arc
        self.radius = None
        
        ## @var contains_radius
        # True if the current line contains a R word
        self.contains_radius = False
        
        ## @var current_spindle_speed
        # Contains the current spindle speed (S word)
        self.current_spindle_speed = None
        
        ## @var contains_spindle
        # True if the current line contains the S word
        self.contains_spindle = False
        
        ## @var dist
        # Distance that current line will travel
        self.dist = 0 
        
        ## @var dists
        # Distance that current line will travel, in [x,y,z] directions
        self.dist_xyz = [0, 0, 0]
        
        ## @var line_is_unsupported_cmd
        # True if current line is not in the whitelist
        self.line_is_unsupported_cmd = False
        
        self.comment = ""
        
        self.special_comment_prefix = "_gcm"
        
        # precompile regular expressions
        self._axes_regexps = []
        self._axes_words = ["X", "Y", "Z"]
        for i in range(0, 3):
            word = self._axes_words[i]
            self._axes_regexps.append(re.compile(".*" + word + "([-.\d]+)"))
            
        self._offset_regexps = []
        self._offset_words = ["I", "J", "K"]
        for i in range(0, 3):
            word = self._offset_words[i]
            self._offset_regexps.append(re.compile(".*" + word + "([-.\d]+)"))
            
        self._re_radius = re.compile(".*R([-.\d]+)")
        
        self._re_use_var = re.compile("#(\d*)")
        self._re_set_var = re.compile("^\s*#(\d+)=([\d.-]+)")
        
        self._re_feed = re.compile(".*F([.\d]+)")
        self._re_feed_replace = re.compile(r"F[.\d]+")
        
        self._re_spindle = re.compile(".*S([\d]+)")
        self._re_spindle_replace = re.compile(r"S[.\d]+")
        
        self._re_motion_mode = re.compile("G([0123])*([^\\d]|$)")
        self._re_distance_mode = re.compile("(G9[01])([^\d]|$)")
        self._re_plane_mode = re.compile("(G1[789])([^\d]|$)")
        self._re_cs = re.compile("(G5[4-9])")
        
        self._re_comment_paren_convert = re.compile("(.*)\((.*?)\)\s*$")
        self._re_comment_paren_replace = re.compile(r"\(.*?\)")
        self._re_comment_get_comment = re.compile("(.*?)(;.*)")
        
        self._re_match_cmd_number = re.compile("([GMT])(\d+)")
        self._re_expand_multicommands = re.compile("([GMT])")


        ## @var whitelist_commands
        # Strip line from everything that is not in this list
        self.whitelist_commands = {
            "G": [
                # non-modal commands
                4,  # Dwell
                10, # set coordintate system
                28, # Go to predefined position
                30, # Go to predefined position
                53, # move in machine coordinates
                92, # set coordintate system offset
                
                # motion modes
                0, # fast linear
                1, # slow linear
                2, # slow arc CW
                3, # slow arc CCW
                38, # probe
                80, # canned cycle stop
                
                # feed rate modes
                93, # inverse
                94, # normal
                
                # units
                20, # inch
                21, # mm
                
                # distance modes
                90, # absulute
                91, # incremental
                
                # plane select
                17, # XY
                18, # ZY
                19, # YZ
                
                # tool length offset
                43, # set compensation
                49, # cancel compensation
                
                # coordinate systems
                54,
                55,
                56,
                57,
                58,
                59,
                ],
            
            "M": [
                # program flow
                0,
                1,
                2,
                30,
                
                # coolant
                8,
                9,
                
                # spindle
                3,
                4,
                5,
                ]
            }
        
        self.logger.info("Preprocessor Class Initialized")
        
        
    def reset(self):
        """
        Reset to initial state.
        """
        self.vars = {}
        self.current_feed = None
        self.current_motion_mode = 0
        self.current_distance_mode = "G90"
        self.callback("on_feed_change", self.current_feed)
        
    @property
    def position_m(self):
        return list(self.pos_m)
    
    @position_m.setter
    def position_m(self, pos):
        self.pos_m = list(pos)
        self.pos_w = list(np.subtract(self.pos_m, self.cs_offsets[self.cs]))
        
    @property
    def current_cs(self):
        return self.cs
    
    @current_cs.setter
    def current_cs(self, label):
        self.cs = label
        self.pos_w = list(np.subtract(self.pos_m, self.cs_offsets[self.cs]))
        
        
    def set_line(self, line):
        """
        Load a Gcode line into the machine. It will be available in `self.line`.
        
        @param line
        A string of Gcode.
        """
        self.line = line
        self.transform_comments()

        
    def split_lines(self):
        """
        Some Gcode generating software spit out 'composite' commands
        like "M3 T2". This really is bad practice.
        
        This method splits such 'composite' commands up into separate
        commands and returns a list of commands. Machine state is not
        changed.
        """
        commands = re.sub(self._re_expand_multicommands, "\n\g<0>", self.line).strip()
        lines = commands.split("\n")
        lines[0] = lines[0] + self.comment # preserve comment
        return lines
    
    
    def strip(self):
        """
        Remove blank spaces and newlines from beginning and end, and remove blank spaces from the middle of the line.
        """
        self.line = self.line.replace(" ", "")
        self.line = self.line.strip()
        
        
    def tidy(self):
        """
        Strips G-Code not contained in the whitelist.
        """
        
        # transform [MG]\d to G\d\d for better parsing
        def format_cmd_number(matchobj):
            cmd = matchobj.group(1)
            cmd_nr = int(matchobj.group(2))
            self.line_is_unsupported_cmd = not (cmd in self.whitelist_commands and cmd_nr in self.whitelist_commands[cmd])
            return "{}{:02d}".format(cmd, cmd_nr)
        
        self.line = re.sub(self._re_match_cmd_number, format_cmd_number, self.line)
        
        if self.line_is_unsupported_cmd:
            self.line = ";" + self.line + " ;" + self.special_comment_prefix + ".unsupported"
    
    
    def parse_state(self):
        """
        This method...
        
        * parses motion mode
        * parses distance mode
        * parses plane mode
        * parses feed rate
        * parses spindle speed
        * parses arcs (offsets and radi)
        * calculates travel distances
        
        ... and updates the machine's state accordingly.
        """
    
        # parse G0 .. G3 and remember
        m = re.match(self._re_motion_mode, self.line)
        if m: self.current_motion_mode = int(m.group(1))
        
        # parse G90 and G91 and remember
        m = re.match(self._re_distance_mode, self.line)
        if m: self.current_distance_mode = m.group(1)
            
        # parse G17, G18 and G19 and remember
        m = re.match(self._re_plane_mode, self.line)
        if m: self.current_plane_mode = m.group(1)
        
        m = re.match(self._re_cs, self.line)
        if m: self.current_cs = m.group(1)
            
        # see if current line has F
        m = re.match(self._re_feed, self.line)
        self.contains_feed = True if m else False
        if m: self.feed_in_current_line = float(m.group(1))
        
        # look for spindle S
        m = re.match(self._re_spindle, self.line)
        self.contains_spindle = True if m else False
        if m: self.current_spindle_speed = int(m.group(1))
        
        # arc parsing and calculations
        if self.current_motion_mode == 2 or self.current_motion_mode == 3:
            self.offset = [None, None, None]
            for i in range(0, 3):
                # loop over I, J, K offsets
                regexp = self._offset_regexps[i]
                
                m = re.match(regexp, self.line)
                if m: self.offset[i] = float(m.group(1))
                    
            # parses arcs
            m = re.match(self._re_radius, self.line)
            self.contains_radius = True if m else False
            if m: self.radius = float(m.group(1))
                
                
        # calculate distance traveled by this G-Code cmd in xyz
        self.dist_xyz = [0, 0, 0] 
        for i in range(0, 3):
            # loop over X, Y, Z axes
            regexp = self._axes_regexps[i]
            
            m = re.match(regexp, self.line)
            if m:
                if self.current_distance_mode == "G90":
                    # absolute distances
                    self.target_m[i] = self.cs_offsets[self.cs][i] + float(m.group(1))
                    self.target_w[i] = float(m.group(1))
                    
                    # calculate distance
                    self.dist_xyz[i] = self.target_m[i] - self.pos_m[i]
                else:
                    # G91 relative distances
                    self.dist_xyz[i] = float(m.group(1))
                    self.target_m[i] += self.dist_xyz[i]
                    self.target_w[i] += self.dist_xyz[i]
            else:
                # no movement along this axis, stays the same
                self.target_m[i] = self.pos_m[i]
                self.target_w[i] = self.pos_w[i]
                    
        # calculate travelling distance
        self.dist = math.sqrt(self.dist_xyz[0] * self.dist_xyz[0] + self.dist_xyz[1] * self.dist_xyz[1] + self.dist_xyz[2] * self.dist_xyz[2])
        
        
    
    
    def fractionize(self):
        """
        Breaks lines longer than a certain threshold into shorter segments.
        
        Also breaks circles into segments.
        
        Returns a list of command strings. Does not update the machine state.
        """

        result = []

        if self.do_fractionize_lines == True and self.current_motion_mode == 1 and self.dist > self.fract_linear_threshold:
            result = self._fractionize_linear_motion()
           
        elif self.do_fractionize_arcs == True and (self.current_motion_mode == 2 or self.current_motion_mode == 3):
            result = self._fractionize_circular_motion()
            
        else:
            # this motion cannot be fractionized
            # return the line as it was passed in
            result = [self.line]

        result[0] = result[0] + self.comment # preserve comment
        return result
        
    
    def done(self):
        """
        When all processing/inspecting of a command has been done, call this method.
        This will virtually 'move' the tool of the machine if the current command
        is a motion command.
        """
        if not (self.current_motion_mode == 0 or self.current_motion_mode == 1):
            # only G0 and G1 can stay active without re-specifying
            self.current_motion_mode = None 
            
        # move the 'tool'
        for i in range(0, 3):
            # loop over X, Y, Z axes
            if self.target_m[i] != None: # keep state
                self.pos_m[i] = self.target_m[i]
                self.pos_w[i] = self.target_w[i]
                
        # re-add comment
        self.line += self.comment
                
        #print("DONE", self.line, self.pos_m, self.pos_w, self.target)


    def find_vars(self):
        """
        Parses all variables in a G-Code line (#1, #2, etc.) and populates
        the internal `vars` dict with corresponding keys and values
        """
        
        m = re.match(self._re_set_var, self.line)
        if m:
            key = m.group(1)
            val = str(float(m.group(2))) # get rid of extra zeros
            self.vars[key] = val
            self.line = ";" + self.line
        
        # find variable usages
        keys = re.findall(self._re_use_var, self.line)
        for key in keys:
            if not key in self.vars: self.vars[key] = None
        
        
    def substitute_vars(self):
        """
        Substitute a variable with a value from the `vars` dict.
        """
        keys = re.findall(self._re_use_var, self.line)
        
        for key in keys:
            val = None
            if key in self.vars:
                val = self.vars[key]
            
            if val == None:
                self.line = ""
                self.callback("on_var_undefined", key)
                return self.line
            else:
                self.line = self.line.replace("#" + key, str(val))
                self.logger.info("SUBSTITUED VAR #{} -> {}".format(key, val))
    
    
    def scale_spindle(self):
        if self.contains_spindle:
            # strip the original S setting
            self.line = re.sub(self._re_spindle_replace, "", self.line).strip()
            self.line += "S{:d}".format(int(self.current_spindle_speed * self.spindle_factor))
                
        
    def override_feed(self):
        """
        Call this method to
        
        * get a callback when the current command contains an F word
        * 
        """
    
        if self.do_feed_override == False and self.contains_feed:
            # Notify parent app of detected feed in current line (useful for UIs)
            if self.current_feed != self.feed_in_current_line:
                self.callback("on_feed_change", self.feed_in_current_line)
            self.current_feed = self.feed_in_current_line
            
            
        if self.do_feed_override == True and self.request_feed:
           
            if self.contains_feed:
                # strip the original F setting
                self.logger.info("STRIPPING FEED: " + self.line)
                self.line = re.sub(self._re_feed_replace, "", self.line).strip()

            if (self.current_feed != self.request_feed):
                self.line += "F{:0.1f}".format(self.request_feed)
                self.current_feed = self.request_feed
                self.logger.info("OVERRIDING FEED: " + str(self.current_feed))
                self.callback("on_feed_change", self.current_feed)

        

    def transform_comments(self):
        """
        Comments in Gcode can be set with semicolon or parentheses.
        This method transforms parentheses comments to semicolon comments.
        """
        
        # transform () comment at end of line into semicolon comment
        self.line = re.sub(self._re_comment_paren_convert, "\g<1>;\g<2>", self.line)
        
        # remove all in-line () comments
        self.line = re.sub(self._re_comment_paren_replace, "", self.line)

        m = re.match(self._re_comment_get_comment, self.line)
        if m:
            self.line = m.group(1)
            self.comment = m.group(2)
        else:
            self.comment = ""


    def _fractionize_circular_motion(self):
        """
        This function is a direct port of Grbl's C code into Python (gcode.c)
        with slight refactoring for Python by Michael Franzl.
        See https://github.com/grbl/grbl
        
        """
        
        # implies self.current_motion_mode == 2 or self.current_motion_mode == 3
        
        if self.current_plane_mode == "G17":
            axis_0 = 0 # X axis
            axis_1 = 1 # Y axis
            axis_linear = 2 # Z axis
        elif self.current_plane_mode == "G18":
            axis_0 = 2 # Z axis
            axis_1 = 0 # X axis
            axis_linear = 1 # Y axis
        elif self.current_plane_mode == "G19":
            axis_0 = 1 # Y axis
            axis_1 = 2 # Z axis
            axis_linear = 0 # X axis
            
        is_clockwise_arc = True if self.current_motion_mode == 2 else False
            
        # deltas between target and (current) position
        x = self.target_w[axis_0] - self.pos_w[axis_0]
        y = self.target_w[axis_1] - self.pos_w[axis_1]
        
        if self.contains_radius:
            # RADIUS MODE
            # R given, no IJK given, self.offset must be calculated
            
            if tuple(self.target_w) == tuple(self.pos_w):
                self.logger.error("Arc in Radius Mode: Identical start/end {}".format(self.line))
                return [self.line]
            
            h_x2_div_d = 4.0 * self.radius * self.radius - x * x - y * y;
            
            if h_x2_div_d < 0:
                self.logger.error("Arc in Radius Mode: Radius error {}".format(self.line))
                return [self.line]

            # Finish computing h_x2_div_d.
            h_x2_div_d = -math.sqrt(h_x2_div_d) / math.sqrt(x * x + y * y);
            
            if not is_clockwise_arc:
                h_x2_div_d = -h_x2_div_d
    
            if self.radius < 0:
                h_x2_div_d = -h_x2_div_d; 
                self.radius = -self.radius;
                
            self.offset[axis_0] = 0.5*(x-(y*h_x2_div_d))
            self.offset[axis_1] = 0.5*(y+(x*h_x2_div_d))
            
        else:
            # CENTER OFFSET MODE, no R given so must be calculated
            
            if self.offset[axis_0] == None or self.offset[axis_1] == None:
                raise Exception("Arc in Offset Mode: No offsets in plane. {}".format(self.line))
                #self.logger.error("Arc in Offset Mode: No offsets in plane")
                #return [self.line]
            
            # Arc radius from center to target
            x -= self.offset[axis_0]
            y -= self.offset[axis_1]
            target_r = math.sqrt(x * x + y * y)
            
            # Compute arc radius for mc_arc. Defined from current location to center.
            self.radius = math.sqrt(self.offset[axis_0] * self.offset[axis_0] + self.offset[axis_1] * self.offset[axis_1])
            
            # Compute difference between current location and target radii for final error-checks.
            delta_r = math.fabs(target_r - self.radius);
            if delta_r > 0.005:
                if delta_r > 0.5:
                    raise Exception("Arc in Offset Mode: Invalid Target. r={:f} delta_r={:f} {}".format(self.radius, delta_r, self.line))
                    #self.logger.warning("Arc in Offset Mode: Invalid Target. r={:f} delta_r={:f} {}".format(self.radius, delta_r, self.line))
                    #return []
                if delta_r > (0.001 * self.radius):
                    raise Exception("Arc in Offset Mode: Invalid Target. r={:f} delta_r={:f} {}".format(self.radius, delta_r, self.line))
                    #self.logger.warning("Arc in Offset Mode: Invalid Target. r={:f} delta_r={:f} {}".format(self.radius, delta_r, self.line))
                    #return []
        
        #print(self.pos_m, self.target, self.offset, self.radius, axis_0, axis_1, axis_linear, is_clockwise_arc)
        
        #print("MCARC", self.line, self.pos_w, self.target_w, self.offset, self.radius, axis_0, axis_1, axis_linear, is_clockwise_arc)
        
        gcode_list = self._mc_arc(self.pos_w, self.target_w, self.offset, self.radius, axis_0, axis_1, axis_linear, is_clockwise_arc)
        
        return gcode_list
        

    def _mc_arc(self, position, target, offset, radius, axis_0, axis_1, axis_linear, is_clockwise_arc):
        """
        This function is a direct port of Grbl's C code into Python (motion_control.c)
        with slight refactoring for Python by Michael Franzl.
        See https://github.com/grbl/grbl
        """
        
        gcode_list = []
        gcode_list.append(";" + self.special_comment_prefix + ".arc_begin[{}]".format(self.line))
        
        do_restore_distance_mode = False
        if self.current_distance_mode == "G91":
            # it's bad to concatenate many small floating point segments due to accumulating errors
            # each arc will use G90
            do_restore_distance_mode = True
            gcode_list.append("G90")
        
        center_axis0 = position[axis_0] + offset[axis_0]
        center_axis1 = position[axis_1] + offset[axis_1]
        # radius vector from center to current location
        r_axis0 = -offset[axis_0]
        r_axis1 = -offset[axis_1]
        # radius vector from target to center
        rt_axis0 = target[axis_0] - center_axis0
        rt_axis1 = target[axis_1] - center_axis1
        
        angular_travel = math.atan2(r_axis0 * rt_axis1 - r_axis1 * rt_axis0, r_axis0 * rt_axis0 + r_axis1 * rt_axis1)
        
        arc_tolerance = 0.004
        arc_angular_travel_epsilon = 0.0000005
        
        if is_clockwise_arc: # Correct atan2 output per direction
            if angular_travel >= -arc_angular_travel_epsilon: angular_travel -= 2*math.pi
        else:
            if angular_travel <= arc_angular_travel_epsilon: angular_travel += 2*math.pi
            
       
            
        segments = math.floor(math.fabs(0.5 * angular_travel * radius) / math.sqrt(arc_tolerance * (2 * radius - arc_tolerance)))
        
        #print("angular_travel:{:f}, radius:{:f}, arc_tolerance:{:f}, segments:{:d}".format(angular_travel, radius, arc_tolerance, segments))
        
        words = ["X", "Y", "Z"]
        if segments:
            theta_per_segment = angular_travel / segments
            linear_per_segment = (target[axis_linear] - position[axis_linear]) / segments
            
            position_last = list(position)
            for i in range(1, segments):
                cos_Ti = math.cos(i * theta_per_segment);
                sin_Ti = math.sin(i * theta_per_segment);
                r_axis0 = -offset[axis_0] * cos_Ti + offset[axis_1] * sin_Ti;
                r_axis1 = -offset[axis_0] * sin_Ti - offset[axis_1] * cos_Ti;
            
                position[axis_0] = center_axis0 + r_axis0;
                position[axis_1] = center_axis1 + r_axis1;
                position[axis_linear] += linear_per_segment;

                gcodeline = ""
                if i == 1:
                    gcodeline += "G1"
                    
                for a in range(0,3):
                    if position[a] != position_last[a]: # only write changes
                        txt = "{}{:0.3f}".format(words[a], position[a])
                        txt = txt.rstrip("0").rstrip(".")
                        gcodeline += txt
                        position_last[a] = position[a]
                        
                if i == 1:
                    if self.contains_feed: gcodeline += "F{:.1f}".format(self.feed_in_current_line)
                    if self.contains_spindle: gcodeline += "S{:d}".format(self.current_spindle_speed)
                    
                gcode_list.append(gcodeline)
            
            
        
        # make sure we arrive at target
        gcodeline = ""
        if segments <= 1:
            gcodeline += "G1"
        
        for a in range(0,3):
            if target[a] != position[a]:
                txt = "{}{:0.3f}".format(words[a], target[a])
                txt = txt.rstrip("0").rstrip(".")
                gcodeline += txt
           
        if segments <= 1:
            # no segments were rendered (very small arc) so we have to put S and F here
            if self.contains_feed: gcodeline += "F{:.1f}".format(self.feed_in_current_line)
            if self.contains_spindle: gcodeline += "S{:d}".format(self.current_spindle_speed)
                
        gcode_list.append(gcodeline)
        
        if do_restore_distance_mode == True:
          gcode_list.append(self.current_distance_mode)
        
        gcode_list.append(";" + self.special_comment_prefix + ".arc_end")
        
        return gcode_list
    
        
    def _fractionize_linear_motion(self):
        gcode_list = []
        gcode_list.append(";" + self.special_comment_prefix + ".line_begin[{}]".format(self.line))
        
        num_fractions = int(self.dist / self.fract_linear_segment_len)
        
        for k in range(0, num_fractions):
            # render segments
            txt = ""
            if k == 0:
                txt += "G1"
                
            for i in range(0, 3):
                # loop over X, Y, Z axes
                segment_length = self.dist_xyz[i] / num_fractions
                coord_rel = (k + 1) * segment_length
                if self.current_distance_mode == "G90":
                    # absolute distances
                    coord_abs = self.pos_w[i] + coord_rel
                    if coord_rel != 0:
                        # only output for changes
                        txt += "{}{:0.3f}".format(self._axes_words[i], coord_abs)
                        txt = txt.rstrip("0").rstrip(".")
                else:
                    # relative distances
                    txt += "{}{:0.3f}".format(self._axes_words[i], segment_length)
                    txt = txt.rstrip("0").rstrip(".")
                
            if k == 0:
                if self.contains_feed: txt += "F{:.1f}".format(self.feed_in_current_line)
                if self.contains_spindle: txt += "S{:d}".format(self.current_spindle_speed)
                
            
            gcode_list.append(txt)
        
        gcode_list.append(";" + self.special_comment_prefix + ".line_end")
        return gcode_list

        


    def _default_callback(self, status, *args):
        print("PREPROCESSOR DEFAULT CALLBACK", status, args)
