import pygame
import math

# ==========================================
# PART 1: GAME LOGIC (The "Environment")
# ==========================================

class SatellitesGame:
    def __init__(self, headless=False):
        self.headless = headless
        
        # Board Setup
        # Rows 0-8. Widths: 8, 9, 10, 11, 12, 11, 10, 9, 8
        self.row_widths = [8, 9, 10, 11, 12, 11, 10, 9, 8]
        self.grid = {} # Key: (row, col), Value: {'owner': 0/1, 'type': 'tank'/'bot', 'count': int}
        
        # Artefacts
        self.artefacts = [(2,1), (2,8), (4,4), (4,7), (6,1), (6,8)]
        
        # Players: 0 (Red), 1 (Blue)
        # Starting units
        self.add_unit(0, 3, 0, 'bot', 2)
        self.add_unit(0, 4, 0, 'tank', 2)
        
        self.add_unit(8, 3, 1, 'bot', 2)
        self.add_unit(8, 4, 1, 'tank', 2)

        # Satellites
        self.satellites = [
            {'type': 'move_tank', 'charges': 2, 'name': 'Move Tank'},
            {'type': 'move_tank', 'charges': 2, 'name': 'Move Tank'},
            {'type': 'move_bot',  'charges': 2, 'name': 'Move Bot'},
            {'type': 'move_bot',  'charges': 2, 'name': 'Move Bot'},
            {'type': 'add_tank',  'charges': 0, 'name': 'Add Tank'},
            {'type': 'add_bot',   'charges': 0, 'name': 'Add Bot'},
        ]
        
        self.scores = [0, 0]
        self.turn = 0 # Player 0 starts
        
        # Turn State Machine
        self.state = "CHOOSE_SATELLITE" 
        
        self.active_satellite_idx = None
        self.actions_remaining = 0
        self.picked_up_charges = 0
        self.action_type = None
        
        self.selected_hex = None 
        
        # Move Quantity Logic
        self.pending_move_dest = None 
        self.pending_move_max = 0
        self.move_amount_selection = 1

        self.info_message = ""
        if not self.headless:
             self.info_message = "Player 1's Turn: Choose a Satellite"
        self.winner = None
        
        # FIX: Turn Limit
        self.turn_count = 1
        self.MAX_TURNS = 100

    def add_unit(self, r, c, owner, u_type, count):
        if (r, c) not in self.grid:
            self.grid[(r, c)] = {'owner': owner, 'type': u_type, 'count': 0}
        self.grid[(r, c)]['count'] += count

    def get_player_unit_count(self, owner):
        total = 0
        for u in self.grid.values():
            if u['owner'] == owner:
                total += u['count']
        return total

    def get_hex_neighbors(self, r, c):
        directions = []
        # Same row
        directions.append((r, c-1))
        directions.append((r, c+1))
        
        # Row Above (r-1)
        if r > 0:
            if r <= 4: # Row above is narrower (or equal for 4->3)
                directions.append((r-1, c-1)) # Top Left
                directions.append((r-1, c))   # Top Right
            else: # Row above is wider
                directions.append((r-1, c))   # Top Left
                directions.append((r-1, c+1)) # Top Right

        # Row Below (r+1)
        if r < 8:
            if r < 4: # Row below is wider
                directions.append((r+1, c))   # Bot Left
                directions.append((r+1, c+1)) # Bot Right
            else: # Row below is narrower
                directions.append((r+1, c-1)) # Bot Left
                directions.append((r+1, c))   # Bot Right

        valid = []
        for nr, nc in directions:
            if 0 <= nr < 9 and 0 <= nc < self.row_widths[nr]:
                valid.append((nr, nc))
        return valid

    def check_actions_still_possible(self):
        """Checks if any valid moves remain for the current action type. If not, auto-end turn."""
        if not self.action_type:
            self.end_turn()
            return

        req_type = 'tank' if 'tank' in self.action_type else 'bot'
        
        can_act = False
        
        # 1. ADD VALID?
        if "add" in self.action_type:
             # Check Cap
             if self.get_player_unit_count(self.turn) >= 20: 
                 can_act = False
             else:
                 # Check Placement Locations
                 if req_type == 'tank':
                     # NEW RULE: Tanks can go on empty space EXCEPT opponent start AND artefacts
                     opp_starts = [(8,3), (8,4)] if self.turn == 0 else [(0,3), (0,4)]
                     
                     # Scan grid for any valid empty spot
                     for r in range(9):
                         for c in range(self.row_widths[r]):
                             # Must be no unit, not opp start, AND not artefact
                             if (r,c) not in self.grid and \
                                (r,c) not in opp_starts and \
                                (r,c) not in self.artefacts:
                                 can_act = True
                                 break
                         if can_act: break

                 else:
                     # OLD RULE (Bots): Start Zones or Own Stacks
                     # 1. Existing Own Stacks?
                     for u in self.grid.values():
                         if u['owner'] == self.turn and u['type'] == req_type:
                            can_act = True
                            break
                     # 2. Empty Start Zones?
                     if not can_act:
                        starts = [(0,3), (0,4)] if self.turn == 0 else [(8,3), (8,4)]
                        for pos in starts:
                            if pos not in self.grid: 
                                can_act = True
                                break
        
        # 2. MOVE VALID?
        elif "move" in self.action_type:
            # Check if user has ANY units of this type that can move
            opp_starts = [(8,3), (8,4)] if self.turn == 0 else [(0,3), (0,4)]
            for pos, unit in self.grid.items():
                if unit['owner'] == self.turn and unit['type'] == req_type:
                    # Check neighbors for THIS unit
                    neighbors = self.get_hex_neighbors(pos[0], pos[1])
                    for nr, nc in neighbors:
                        # NEW RULE: No entry to opponent starting hexes
                        if (nr, nc) in opp_starts: continue

                        target_cell = self.grid.get((nr,nc))
                        
                        # Apply same rules as in action_mask
                        # 1. Tank -> Artefact = No
                        if req_type == 'tank' and (nr, nc) in self.artefacts: continue
                        
                        # 2. Bot -> Enemy = No
                        if req_type == 'bot' and target_cell and target_cell['owner'] != self.turn: continue
                        
                        # 3. Diff Type Merge = No
                        if target_cell and target_cell['owner'] == self.turn and target_cell['type'] != req_type: continue
                        
                        # 4. Tank Attack Size Rule
                        if req_type == 'tank' and target_cell and target_cell['owner'] != self.turn:
                            if target_cell['type'] == 'tank' and target_cell['count'] >= unit['count']:
                                continue
                        
                        # If we reach here, at least one move is possible
                        can_act = True
                        break
                if can_act: break
        
        if not can_act:
            self.end_turn()
            if not self.headless:
                self.info_message = f"Skipped: No valid {self.action_type} actions."

    def execute_add(self, r, c):
        # 1. SECURITY CHECK: State
        if self.state != "PERFORM_ACTIONS": return False
        if "add" not in self.action_type: return False

        unit_type = 'tank' if 'tank' in self.action_type else 'bot'
        current = self.grid.get((r,c))
        
        # 2. SECURITY CHECK: Unit Cap
        current_count = self.get_player_unit_count(self.turn)
        if current_count >= 20:
             self.info_message = "Unit Cap Reached (20 Max)!"
             return False

        # 3. SECURITY CHECK: Valid Placement Location
        
        if unit_type == 'tank':
            # === NEW TANK RULE ===
            
            # 1. Must be strictly empty (No merging with own tanks)
            if current is not None:
                self.info_message = "Tanks: Must be placed on empty space."
                return False

            # 2. Must not be opponent start zone
            opp_starts = [(8,3), (8,4)] if self.turn == 0 else [(0,3), (0,4)]
            if (r,c) in opp_starts:
                self.info_message = "Cannot place in opponent start zone."
                return False

            # 3. Must not be an Artefact
            if (r,c) in self.artefacts:
                self.info_message = "Cannot place tank on an Artefact."
                return False
                
            # === EXECUTION (Single Unit) ===
            self.grid[(r,c)] = {'owner': self.turn, 'type': 'tank', 'count': 1}
            self.actions_remaining -= 1
            self.info_message = f"Added tank. Actions: {self.actions_remaining}"
            
            if self.actions_remaining <= 0:
                self.end_turn()
            else:
                self.check_actions_still_possible()
            return True

        else:
            # === OLD BOT RULE (Unchanged) ===
            # 1. Own Stacks (Merging)
            is_own_stack = (current and current['owner'] == self.turn and current['type'] == unit_type)
            
            # 2. Starting Zones (if empty or own)
            valid_starts = [(0,3), (0,4)] if self.turn == 0 else [(8,3), (8,4)]
            is_start_zone = ((r,c) in valid_starts) and (not current or is_own_stack)

            if not (is_own_stack or is_start_zone):
                self.info_message = "Bots: Drop on Start Zone or Own Stack"
                return False

            # --- EXECUTION ---
            if current:
                current['count'] += 1
                self.actions_remaining -= 1
                self.info_message = f"Added {unit_type}. Actions: {self.actions_remaining}"
            else:
                self.grid[(r,c)] = {'owner': self.turn, 'type': unit_type, 'count': 1}
                self.actions_remaining -= 1
                self.info_message = f"Added {unit_type}. Actions: {self.actions_remaining}"
                
            if self.actions_remaining <= 0:
                self.end_turn()
            else:
                self.check_actions_still_possible()
            
            return True

    def handle_click(self, r, c):
        if self.state != "PERFORM_ACTIONS": return
        
        # 1. HANDLE ADDING UNITS
        if "add" in self.action_type:
            # We simply try to add at the clicked location.
            # The core engine will reject it if it's not a start zone or valid stack.
            success = self.execute_add(r, c)
            if not success and self.headless:
                print(f"Add rejected by core: {self.info_message}")

        # 2. HANDLE MOVING UNITS
        elif "move" in self.action_type:
            if self.selected_hex:
                # User has already selected a source and is clicking a destination
                
                # Deselect if clicking the same hex
                if (r,c) == self.selected_hex:
                    self.selected_hex = None 
                    return

                # --- THE "DUMB" UI CHANGE ---
                # Previously, we checked for neighbors, artefacts, and unit types here.
                # NOW, we remove all those checks. We just ask the engine:
                # "Can I move from Selected to Here?"
                
                success, _, _ = self.execute_move(self.selected_hex, (r,c), self.move_amount_selection)
                if not success:
                    # If the move failed (illegal), we usually keep the selection
                    # so the user can try a valid move instead.
                    pass 
                else:
                    # If successful, the engine has already updated the grid.
                    pass

            else:
                # User is selecting the Source hex
                if (r,c) in self.grid and self.grid[(r,c)]['owner'] == self.turn:
                    req_type = 'tank' if 'tank' in self.action_type else 'bot'
                    
                    # We still check if the unit type matches the satellite card
                    # (This is a basic UI usability filter, not really a rule check)
                    if self.grid[(r,c)]['type'] == req_type:
                        self.selected_hex = (r,c)
                        self.move_amount_selection = self.grid[(r,c)]['count']
                        self.info_message = f"Selected {req_type}. Click destination."
                    else:
                        self.info_message = f"Wrong unit type! Satellite needs {req_type}."

    def check_win(self):
        # 1. Score >= 9
        if self.scores[self.turn] >= 10:
            self.winner = self.turn
            self.state = "GAME_OVER"
            return True
        # 2. All Artefacts Captured
        if len(self.artefacts) == 0:
            if self.scores[0] > self.scores[1]: self.winner = 0
            elif self.scores[1] > self.scores[0]: self.winner = 1
            else: self.winner = self.turn # Tie-breaker
            self.state = "GAME_OVER"
            return True
        return False

   # In SatellitesGame class

    # In SatellitesGame class

    def execute_move(self, start, end, amount):
        # 1. SECURITY CHECKS
        # Failures must now return a tuple: (False, 0, 0)
        # 0 kills, 0 points
        if start not in self.grid: return False, 0, 0
        cell = self.grid[start]
        if cell['owner'] != self.turn: return False, 0, 0
        
        neighbors = self.get_hex_neighbors(start[0], start[1])
        if end not in neighbors:
            self.info_message = "Invalid Move: Not adjacent"
            return False, 0, 0

        opp_starts = [(8,3), (8,4)] if self.turn == 0 else [(0,3), (0,4)]
        if end in opp_starts:
            self.info_message = "Cannot move onto opponent starting hex!"
            return False, 0, 0

        move_type = cell['type']
        if move_type == 'tank' and end in self.artefacts:
            self.info_message = "Tanks cannot capture artefacts!"
            return False, 0, 0

        # --- EXECUTION ---
        cell['count'] -= amount
        if cell['count'] < 0: return False, 0, 0
        
        if cell['count'] == 0:
            del self.grid[start]
            
        target = self.grid.get(end)
        did_move_in = True 
        
        # Track our rewards
        units_destroyed = 0 
        score_gain = 0
        
        if target:
            if target['owner'] == self.turn:
                # Merge
                if target['type'] == move_type:
                    target['count'] += amount
                else: 
                     # Blocked
                     self.info_message = "Blocked: Different unit type"
                     if start not in self.grid: self.grid[start] = cell
                     self.grid[start]['count'] += amount 
                     return False, 0, 0
            else:
                # Combat / Attack
                if move_type == 'bot':
                    self.info_message = "Bots cannot attack!"
                    if start not in self.grid: self.grid[start] = cell
                    self.grid[start]['count'] += amount
                    return False, 0, 0

                if move_type == 'tank' and target['type'] == 'tank' and target['count'] >= amount:
                     self.info_message = "Can only attack smaller tank stack!"
                     if start not in self.grid: self.grid[start] = cell
                     self.grid[start]['count'] += amount
                     return False, 0, 0

                # Successful Kill
                units_destroyed = target['count'] 
                del self.grid[end]
                
                if move_type == 'tank':
                    did_move_in = False
                    if start not in self.grid: self.grid[start] = cell
                    self.grid[start]['count'] += amount
                    self.info_message = "Attack Successful! Tank holds position."
                else:
                    self.grid[end] = {'owner': self.turn, 'type': move_type, 'count': amount}
        else:
            # Move to empty
            self.grid[end] = {'owner': self.turn, 'type': move_type, 'count': amount}
        
        # --- ARTEFACT LOGIC ---
        if did_move_in and end in self.artefacts:
            self.artefacts.remove(end)
            # Rule: 1 point per bot in the stack
            score_gain = amount  
            self.scores[self.turn] += score_gain 
            self.info_message = f"Captured Artefact! +{score_gain} pts"

        if self.check_win():
             return True, units_destroyed, score_gain

        self.actions_remaining -= 1
        
        # Message Logic
        if did_move_in and "Captured" not in self.info_message:
            self.info_message = f"Moved {amount} units. Actions: {self.actions_remaining}"
        elif not did_move_in:
            self.info_message += f" ({self.actions_remaining} left)"
        
        if self.actions_remaining <= 0:
            self.end_turn()
            self.selected_hex = None
        else:
            if did_move_in:
                self.selected_hex = end
            else:
                self.selected_hex = start
            
        return True, units_destroyed, score_gain

    def select_satellite(self, idx):
        if self.state != "CHOOSE_SATELLITE": return
        sat = self.satellites[idx]
        if sat['charges'] > 0:
            self.active_satellite_idx = idx
            self.action_type = sat['type']
            self.picked_up_charges = sat['charges']
            
            # Remove charges immediately
            sat['charges'] = 0
            
            # NEW SATE: Choose Direction
            self.state = "CHOOSE_DIRECTION"
            self.info_message = "Choose Distribution Direction"
        else:
            self.info_message = "Satellite expects charges!"

    def set_distribution_direction(self, clockwise):
        self.distribution_direction = 1 if clockwise else -1
        
        # Distribute Immediately
        self.perform_distribution()
        
        # FIX: Check if action is possible
        req_type = 'tank' if 'tank' in self.satellites[self.active_satellite_idx]['type'] else 'bot'
        action_main = 'move' if 'move' in self.satellites[self.active_satellite_idx]['type'] else 'add'
        
        can_act = True
        if action_main == 'move':
            # Check if user has ANY units of this type
            has_units = False
            for u in self.grid.values():
                if u['owner'] == self.turn and u['type'] == req_type:
                    has_units = True
                    break
            if not has_units:
                can_act = False
                self.info_message = f"No {req_type}s to move! Turn Ending."
        
        elif action_main == 'add':
            # Check if any valid placement exists
            can_add = False
            
            # 1. Check Cap
            if self.get_player_unit_count(self.turn) >= 20: 
                can_add = False # At cap, no adds allowed
            else:
                if req_type == 'tank':
                    # NEW RULE: Any empty space OR Own Tank
                    # Assume true unless grid is 100% full of enemies/wrong types
                    # Simple check: is grid full?
                    total_hexes = sum(self.row_widths)
                    if len(self.grid) < total_hexes:
                        can_add = True
                    else:
                        for u in self.grid.values():
                            if u['owner'] == self.turn and u['type'] == 'tank':
                                can_add = True
                                break
                else:
                    # OLD RULE (Bots)
                    # 2. Existing Own Stacks?
                    for u in self.grid.values():
                        if u['owner'] == self.turn and u['type'] == req_type:
                            can_add = True
                            break
                    
                    # 3. Empty Start Zones?
                    if not can_add:
                        starts = [(0,3), (0,4)] if self.turn == 0 else [(8,3), (8,4)]
                        for pos in starts:
                            if pos not in self.grid: # Empty start zone
                                can_add = True
                                break
            
            if not can_add:
                can_act = False
                self.info_message = f"No valid placement for {req_type}! Turn Ending."
        
        if can_act:
            self.actions_remaining = self.picked_up_charges
            self.state = "PERFORM_ACTIONS"
            self.info_message = f"Action: {self.satellites[self.active_satellite_idx]['name']} ({self.actions_remaining} remaining)"
        else:
            self.end_turn()
            self.info_message = f"Skipped Reason: No {req_type}s."

    def perform_distribution(self):
        # Use stored direction
        to_distribute = self.picked_up_charges 
        idx = self.active_satellite_idx
        
        while to_distribute > 0:
            idx = (idx + self.distribution_direction) % 6
            self.satellites[idx]['charges'] += 1
            to_distribute -= 1

    def end_turn(self):
        # FIX: Check Turn Limit
        if self.turn_count >= self.MAX_TURNS:
            self.state = "GAME_OVER"
            self.info_message = "Max Turn Limit Reached."
            if self.scores[0] > self.scores[1]: self.winner = 0
            elif self.scores[1] > self.scores[0]: self.winner = 1
            else: self.winner = -1 # Draw
            return

        self.turn = 1 - self.turn
        if self.turn == 0: 
            self.turn_count += 1
            # Log turn if needed or other round-based logic
        
        self.state = "CHOOSE_SATELLITE"
        self.selected_hex = None
        self.active_satellite_idx = None
        
        p_name = "Player 1 (Blue)" if self.turn == 1 else "Player 0 (Red)"
        if not self.headless and "Skipped" not in self.info_message:
            self.info_message = f"{p_name}'s Turn. Choose Satellite."


# ==========================================
# PART 2: PYGAME UI
# ==========================================

class SatellitesUI:
    def __init__(self, game):
        self.game = game
        pygame.init()
        self.width = 1000
        self.height = 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Satellites Game")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 20)
        self.title_font = pygame.font.SysFont('Arial', 32, bold=True)
        self.huge_font = pygame.font.SysFont('Arial', 48, bold=True)
        self.score_font = pygame.font.SysFont('Arial', 40, bold=True)
        self.game_over_font = pygame.font.SysFont('Arial', 80, bold=True)
        
        # Geometry
        self.hex_radius = 25
        self.hex_height = math.sqrt(3) * self.hex_radius
        self.hex_width = 2 * self.hex_radius
        self.board_center_x = self.width // 2
        self.board_center_y = 350
        
        # Precomputed grid centers
        self.hex_centers = {}
        for r in range(9):
            row_w = game.row_widths[r]
            y = self.board_center_y + (r - 4) * (self.hex_height * 0.9) # overlap slightly
            start_x = self.board_center_x - (row_w * self.hex_width / 2) + (self.hex_radius)
            for c in range(row_w):
                x = start_x + c * (self.hex_width * 1.0) 
                self.hex_centers[(r,c)] = (int(x), int(y))

    def hex_corners(self, center, radius):
        points = []
        for i in range(6):
            angle_deg = 60 * i + 30 
            angle_rad = math.radians(angle_deg)
            points.append((
                center[0] + radius * math.cos(angle_rad),
                center[1] + radius * math.sin(angle_rad)
            ))
        return points

    def draw_button(self, rect, text, color, text_color=(255,255,255)):
        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, (255,255,255), rect, 2)
        txt_surf = self.font.render(text, True, text_color)
        txt_rect = txt_surf.get_rect(center=rect.center)
        self.screen.blit(txt_surf, txt_rect)

    def draw_triangle(self, center, size, color):
        # Pointing up
        half = size / 2
        points = [
            (center[0], center[1] - half),
            (center[0] - half, center[1] + half),
            (center[0] + half, center[1] + half)
        ]
        pygame.draw.polygon(self.screen, color, points)

    def draw(self):
        self.screen.fill((30, 30, 30))
        
        # Top HUD - Turn Indicator
        turn_color = (255, 100, 100) if self.game.turn == 0 else (100, 100, 255)
        pygame.draw.rect(self.screen, (50, 50, 50), (0, 0, self.width, 60))
        turn_text = f"PLAYER {self.game.turn} ({'RED' if self.game.turn == 0 else 'BLUE'}) TURN"
        turn_surf = self.title_font.render(turn_text, True, turn_color)
        self.screen.blit(turn_surf, (self.width//2 - turn_surf.get_width()//2, 10))
        
        # Scores
        score_y = 10
        # P0 Score (Red)
        s0 = self.score_font.render(str(self.game.scores[0]), True, (255, 100, 100))
        self.screen.blit(s0, (20, score_y))
        # P1 Score (Blue)
        s1 = self.score_font.render(str(self.game.scores[1]), True, (100, 100, 255))
        self.screen.blit(s1, (self.width - 20 - s1.get_width(), score_y))
        
        # Info Message
        msg_surf = self.font.render(self.game.info_message, True, (200, 200, 200))
        self.screen.blit(msg_surf, (20, 70))
        
        # Draw Hex Grid
        for r in range(9):
            for c in range(self.game.row_widths[r]):
                center = self.hex_centers[(r,c)]
                poly = self.hex_corners(center, self.hex_radius - 2)
                
                color = (60, 60, 60)
                if self.game.selected_hex == (r,c):
                    color = (150, 150, 50)
                elif self.game.state == "SELECT_MOVE_AMOUNT" and self.game.pending_move_dest == (r,c):
                    color = (150, 50, 150) # Highlight dest
                
                pygame.draw.polygon(self.screen, color, poly)
                pygame.draw.polygon(self.screen, (100, 100, 100), poly, 1)

        # Draw Artefacts
        for (ar, ac) in self.game.artefacts:
            center = self.hex_centers.get((ar, ac))
            if center:
                pygame.draw.circle(self.screen, (255, 215, 0), center, 5)

        # Draw Units
        for (r,c), unit in self.game.grid.items():
            center = self.hex_centers[(r,c)]
            color = (255, 100, 100) if unit['owner'] == 0 else (100, 100, 255)
            if unit['type'] == 'tank':
                # FIX: Draw Triangle for Tank
                self.draw_triangle(center, 20, color)
            else:
                pygame.draw.circle(self.screen, color, center, 10)
            ct_text = self.font.render(str(unit['count']), True, (255,255,255))
            self.screen.blit(ct_text, (center[0]-5, center[1]-25))
            
            # Show move amount selection next to selected hex
            if self.game.selected_hex == (r,c) and "move" in self.game.action_type:
                move_text = self.font.render(f"→ {self.game.move_amount_selection}", True, (255, 255, 0))
                self.screen.blit(move_text, (center[0] + 30, center[1] - 10))

        # Draw Satellites (Cards)
        card_w = 140
        card_h = 100
        start_x = (self.width - (6 * 150)) // 2
        y = self.height - 120
        
        for i, sat in enumerate(self.game.satellites):
            x = start_x + i * 150
            rect = pygame.Rect(x, y, card_w, card_h)
            
            bg_color = (50, 50, 80)
            if self.game.active_satellite_idx == i:
                bg_color = (80, 80, 120)
            
            pygame.draw.rect(self.screen, bg_color, rect)
            pygame.draw.rect(self.screen, (200, 200, 200), rect, 2)
            
            name = self.font.render(sat['name'], True, (255,255,255))
            charges = self.title_font.render(str(sat['charges']), True, (255, 255, 0))
            
            self.screen.blit(name, (x + 10, y + 10))
            self.screen.blit(charges, (x + 60, y + 50))
            
            # FIX: Draw Icon on Card (Triangle for Tank)
            icon_center = (x + card_w - 25, y + 25)
            icon_color = (180, 180, 180)
            if 'tank' in sat['type']:
                self.draw_triangle(icon_center, 16, icon_color)
            else:
                pygame.draw.circle(self.screen, icon_color, icon_center, 8)
            
            sat['rect'] = rect

        # POPUPS
        cx, cy = self.width // 2, self.height // 2
        
        # 1. Direction Choice
        if self.game.state == "CHOOSE_DIRECTION":
            # Overlay
            s = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            s.fill((0,0,0,128))
            self.screen.blit(s, (0,0))
            
            self.cw_btn = pygame.Rect(cx + 20, cy - 25, 150, 50)
            self.ccw_btn = pygame.Rect(cx - 170, cy - 25, 150, 50)
            
            self.draw_button(self.cw_btn, "Clockwise >>", (50, 150, 50))
            self.draw_button(self.ccw_btn, "<< Counter-CW", (50, 50, 150))
            
            instr = self.title_font.render("Choose Charge Distribution", True, (255,255,255))
            self.screen.blit(instr, (cx - instr.get_width()//2, cy - 80))
            
            # Add hint for keyboard shortcuts
            hint = self.font.render("(← Left Arrow = Counter-CW   |   Right Arrow → = Clockwise)", True, (180,180,180))
            self.screen.blit(hint, (cx - hint.get_width()//2, cy + 40))

        # 2. Move Quantity Selection
        if self.game.state == "SELECT_MOVE_AMOUNT":
            # Overlay
            s = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            s.fill((0,0,0,128))
            self.screen.blit(s, (0,0))
            
            panel = pygame.Rect(cx - 150, cy - 100, 300, 200)
            pygame.draw.rect(self.screen, (40,40,40), panel)
            pygame.draw.rect(self.screen, (200,200,200), panel, 2)
            
            title = self.font.render(f"Move How Many? (Max {self.game.pending_move_max})", True, (255,255,255))
            self.screen.blit(title, (cx - title.get_width()//2, cy - 80))
            
            num = self.huge_font.render(str(self.game.move_amount_selection), True, (255,255,0))
            self.screen.blit(num, (cx - num.get_width()//2, cy - 20))
            
            self.minus_btn = pygame.Rect(cx - 100, cy - 15, 40, 40)
            self.plus_btn = pygame.Rect(cx + 60, cy - 15, 40, 40)
            self.confirm_btn = pygame.Rect(cx - 60, cy + 50, 120, 40)
            
            self.draw_button(self.minus_btn, "-", (100, 50, 50))
            self.draw_button(self.plus_btn, "+", (50, 100, 50))
            self.draw_button(self.confirm_btn, "MOVE", (50, 50, 150))

        # GAME OVER Overlay
        if self.game.state == "GAME_OVER":
            s = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            s.fill((0,0,0,200))
            self.screen.blit(s, (0,0))
            
            if self.game.winner == -1:
                winner_text = "DRAW!"
                color = (200, 200, 200)
            else:
                winner_text = "RED WINS!" if self.game.winner == 0 else "BLUE WINS!"
                color = (255, 100, 100) if self.game.winner == 0 else (100, 100, 255)
            
            w_surf = self.game_over_font.render(winner_text, True, color)
            self.screen.blit(w_surf, (cx - w_surf.get_width()//2, cy - 50))
            
            reason = self.font.render("Game Over", True, (255,255,255))
            self.screen.blit(reason, (cx - reason.get_width()//2, cy + 20))
        
        pygame.display.flip()

    def run(self):
        running = True
        while running:
            self.clock.tick(60)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                # FIX: Keyboard input
                elif event.type == pygame.KEYDOWN:
                    if self.game.state == "CHOOSE_DIRECTION":
                        if event.key == pygame.K_LEFT:
                            self.game.set_distribution_direction(False)  # Counter-clockwise
                        elif event.key == pygame.K_RIGHT:
                            self.game.set_distribution_direction(True)   # Clockwise
                    
                    elif self.game.state == "PERFORM_ACTIONS" and self.game.selected_hex and "move" in self.game.action_type:
                        # Allow adjusting move amount when a hex is selected
                        max_count = self.game.grid[self.game.selected_hex]['count']
                        if event.key == pygame.K_UP:
                            self.game.move_amount_selection = min(max_count, self.game.move_amount_selection + 1)
                            self.game.info_message = f"Moving {self.game.move_amount_selection} units (adjust with Up/Down)"
                        elif event.key == pygame.K_DOWN:
                            self.game.move_amount_selection = max(1, self.game.move_amount_selection - 1)
                            self.game.info_message = f"Moving {self.game.move_amount_selection} units (adjust with Up/Down)"
                    
                    elif self.game.state == "SELECT_MOVE_AMOUNT":
                        if event.key == pygame.K_UP:
                             self.game.move_amount_selection = min(self.game.pending_move_max, self.game.move_amount_selection + 1)
                        elif event.key == pygame.K_DOWN:
                             self.game.move_amount_selection = max(1, self.game.move_amount_selection - 1)
                        elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                            self.game.execute_move(self.game.selected_hex, self.game.pending_move_dest, self.game.move_amount_selection)
                            if self.game.state == "SELECT_MOVE_AMOUNT":
                                self.game.state = "PERFORM_ACTIONS"
                            self.game.pending_move_dest = None

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = pygame.mouse.get_pos()
                    
                    if self.game.state == "CHOOSE_DIRECTION":
                        if hasattr(self, 'cw_btn') and self.cw_btn.collidepoint(mx, my):
                            self.game.set_distribution_direction(True)
                        elif hasattr(self, 'ccw_btn') and self.ccw_btn.collidepoint(mx, my):
                            self.game.set_distribution_direction(False)
                    
                    elif self.game.state == "SELECT_MOVE_AMOUNT":
                        if hasattr(self, 'minus_btn') and self.minus_btn.collidepoint(mx, my):
                            self.game.move_amount_selection = max(1, self.game.move_amount_selection - 1)
                        elif hasattr(self, 'plus_btn') and self.plus_btn.collidepoint(mx, my):
                            self.game.move_amount_selection = min(self.game.pending_move_max, self.game.move_amount_selection + 1)
                        elif hasattr(self, 'confirm_btn') and self.confirm_btn.collidepoint(mx, my):
                            # Execute
                            success, _, _ = self.game.execute_move(self.game.selected_hex, self.game.pending_move_dest, self.game.move_amount_selection)
                            
                            if self.game.state == "SELECT_MOVE_AMOUNT":
                                self.game.state = "PERFORM_ACTIONS"
                            self.game.pending_move_dest = None
                            
                    elif self.game.state == "CHOOSE_SATELLITE":
                        for i, sat in enumerate(self.game.satellites):
                            if 'rect' in sat and sat['rect'].collidepoint(mx, my):
                                self.game.select_satellite(i)
                                break
                    
                    elif self.game.state == "PERFORM_ACTIONS":
                        best_dist = 9999
                        best_hex = None
                        for (r,c), center in self.hex_centers.items():
                            dist = math.hypot(mx - center[0], my - center[1])
                            if dist < self.hex_radius and dist < best_dist:
                                best_dist = dist
                                best_hex = (r,c)
                        
                        if best_hex:
                            self.game.handle_click(best_hex[0], best_hex[1])
            
            self.draw()
            
        pygame.quit()

if __name__ == "__main__":
    game = SatellitesGame()
    ui = SatellitesUI(game)
    ui.run()