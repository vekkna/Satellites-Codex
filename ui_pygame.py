import math
import pygame

from agents.mcts import MCTS, SatellitesAdapter
from engine import SatellitesGame

# ==========================================
# PART 2: PYGAME UI
# ==========================================

class SatellitesUI:
    def __init__(self, game):
        self.game = game
        self.ai_adapter = SatellitesAdapter()
        self.mcts = MCTS(self.ai_adapter, iterations=120, rollout_depth=16, seed=1)
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

        self.show_weights_panel = False
        self.player_control = {0: "human", 1: "human"}
        self.ai_move_cooldown_ms = 80
        self.last_ai_move_ms = 0
        self.weight_rows = [
            ("Win score", "score_diff", 5.0, 0.0, 300.0),
            ("Kill 3+ bots", "move_tank_adj_enemy_bot", 0.2, 0.0, 8.0),
            ("Kill bots @ artefact", "move_bot_capture", 0.5, 0.0, 12.0),
            ("Kill smaller tank", "move_tank_vs_tank_win", 0.2, 0.0, 8.0),
            ("Block lanes", "add_tank_near_artefact", 0.2, 0.0, 8.0),
            ("Reinforce near tank", "sat_add_tank_bonus", 0.2, 0.0, 8.0),
        ]
        self.weight_buttons = []

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
        mode_text = (
            f"P0:{self.player_control[0].upper()}  "
            f"P1:{self.player_control[1].upper()}  "
            f"[1]/[2] Toggle   [T] Toggle current   [W] Weights"
        )
        mode_surf = self.font.render(mode_text, True, (180, 180, 180))
        self.screen.blit(mode_surf, (20, 95))
        
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

        if self.show_weights_panel:
            self.draw_weights_panel()
        
        pygame.display.flip()

    def is_ai_turn(self):
        if self.game.state == "GAME_OVER":
            return False
        return self.player_control.get(self.game.turn, "human") == "ai"

    def toggle_player_control(self, player):
        cur = self.player_control.get(player, "human")
        self.player_control[player] = "ai" if cur == "human" else "human"

    def maybe_run_ai_turn(self):
        if not self.is_ai_turn():
            return
        now = pygame.time.get_ticks()
        if now - self.last_ai_move_ms < self.ai_move_cooldown_ms:
            return
        self.last_ai_move_ms = now
        try:
            action, _ = self.mcts.select_action(self.game)
            ok = self.game.apply_action(action)
            if not ok:
                self.game.info_message = f"AI chose illegal action: {action}"
        except ValueError:
            # No legal actions available
            pass

    def draw_weights_panel(self):
        panel = pygame.Rect(self.width - 300, 80, 285, 280)
        pygame.draw.rect(self.screen, (25, 25, 25), panel)
        pygame.draw.rect(self.screen, (200, 200, 200), panel, 2)
        title = self.font.render("AI Weights (W to hide)", True, (255, 255, 255))
        self.screen.blit(title, (panel.x + 10, panel.y + 8))

        self.weight_buttons = []
        weights = self.ai_adapter.get_weights()
        y = panel.y + 40
        for label, key, step, lo, hi in self.weight_rows:
            minus_rect = pygame.Rect(panel.x + 8, y + 2, 24, 20)
            plus_rect = pygame.Rect(panel.x + 250, y + 2, 24, 20)
            value = weights.get(key, 0.0)

            self.draw_button(minus_rect, "-", (100, 50, 50))
            self.draw_button(plus_rect, "+", (50, 100, 50))
            line = self.font.render(f"{label}: {value:.2f}", True, (220, 220, 220))
            self.screen.blit(line, (panel.x + 40, y))
            self.weight_buttons.append((minus_rect, plus_rect, key, step, lo, hi))
            y += 38

    def run(self):
        running = True
        while running:
            self.clock.tick(60)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                # FIX: Keyboard input
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_w:
                        self.show_weights_panel = not self.show_weights_panel
                    elif event.key == pygame.K_1:
                        self.toggle_player_control(0)
                    elif event.key == pygame.K_2:
                        self.toggle_player_control(1)
                    elif event.key == pygame.K_t:
                        self.toggle_player_control(self.game.turn)
                    if self.is_ai_turn():
                        continue
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
                    if self.show_weights_panel and self.handle_weights_click(mx, my):
                        continue
                    if self.is_ai_turn():
                        continue
                    
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
            
            self.maybe_run_ai_turn()
            self.draw()
            
        pygame.quit()

    def handle_weights_click(self, mx, my):
        for minus_rect, plus_rect, key, step, lo, hi in self.weight_buttons:
            if minus_rect.collidepoint(mx, my):
                cur = self.ai_adapter.get_weights().get(key, 0.0)
                self.ai_adapter.set_weight(key, max(lo, cur - step))
                return True
            if plus_rect.collidepoint(mx, my):
                cur = self.ai_adapter.get_weights().get(key, 0.0)
                self.ai_adapter.set_weight(key, min(hi, cur + step))
                return True
        return False

def main():
    game = SatellitesGame()
    ui = SatellitesUI(game)
    ui.run()

if __name__ == "__main__":
    main()

