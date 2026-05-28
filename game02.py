import pygame
import math
import sys
import heapq # A* 알고리즘용 우선순위 큐 모듈

# --- [1. 설정 값] ---
pygame.init()
WIDTH, HEIGHT = 1000, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Auto Drive - Perfect APF & A* Integration")
clock = pygame.time.Clock()

WORLD_SCALE = 10
WHITE, BLACK, GRAY = (255, 255, 255), (0, 0, 0), (200, 200, 200)
GREEN, RED, BLUE, YELLOW = (34, 139, 34), (255, 0, 0), (0, 120, 215), (255, 220, 0)
HOVER_BLUE = (0, 180, 255)
PURPLE = (155, 89, 182) # 가상 원형 장애물을 표현할 보라색 색상
PATH_COLOR = (255, 105, 180) # 웨이포인트(네비게이션 핑크선) 색상
FONT_L = pygame.font.SysFont("malgungothic", 50, bold=True)
FONT_S = pygame.font.SysFont("malgungothic", 20)
FONT_XS = pygame.font.SysFont("malgungothic", 16, bold=True)

# --- [추가 보조 함수: 장애물의 무게중심과 완전히 덮는 반지름 구하기] ---
def get_virtual_circle(obs):
    if not obs or len(obs) == 0: 
        return (0, 0), 0
    sum_x = sum(p[0] for p in obs)
    sum_y = sum(p[1] for p in obs)
    gx, gy = sum_x / len(obs), sum_y / len(obs)
    
    max_r = 0
    for p in obs:
        r = math.hypot(p[0] - gx, p[1] - gy)
        if r > max_r: max_r = r
    return (gx, gy), max_r

# --- [2. 자동차 클래스] ---
class Car:
    def __init__(self, x, y, goal_pos):
        self.x, self.y = float(x), float(y)
        self.goal_pos = goal_pos
        self.angle = 0
        self.speed = 0.0
        self.acceleration = 0.2
        self.friction_val = 14 
        
        self.max_speed_limit = 150 
        self.target_val = 60       
        self.ai_enabled = False    
        
        self.max_hp = 500.0
        self.hp = self.max_hp
        self.waypoints = [] 
        
        self.width, self.height = 40, 70
        self.original_img = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.rect(self.original_img, RED, (0, 0, self.width, self.height), border_radius=8)
        pygame.draw.rect(self.original_img, BLACK, (5, 10, self.width-10, 20)) 

    # --- [A* 경로 탐색 함수] ---
    def recalculate_path(self, obstacles):
        node_size = 120
        start_node = (int(self.x // node_size), int(self.y // node_size))
        goal_node = (int(self.goal_pos[0] // node_size), int(self.goal_pos[1] // node_size))

        def is_blocked(nx, ny):
            cx, cy = nx * node_size + node_size/2, ny * node_size + node_size/2
            corners = [
                (cx - node_size/2, cy - node_size/2), (cx + node_size/2, cy - node_size/2),
                (cx + node_size/2, cy + node_size/2), (cx - node_size/2, cy + node_size/2)
            ]
            for obs in obstacles:
                # [핵심 추가] 1. 보라색 가상 원형 장애물(O') 회피 인식
                (gx, gy), r = get_virtual_circle(obs)
                dist_to_g = math.hypot(cx - gx, cy - gy)
                # 원의 반지름(r)에 여유 마진(40)을 두어 A*가 보라색 원 바깥으로 길을 그리게 유도
                if dist_to_g < (r + 40): 
                    return True

                # 2. 실제 노란색 장애물 선분 충돌 검사
                for i in range(len(obs)-1):
                    if dist_to_line_points(corners, obs[i], obs[i+1]): return True
                    if dist_to_line((cx, cy), obs[i], obs[i+1]) < node_size * 0.6: return True
            return False

        open_set = []
        heapq.heappush(open_set, (0, start_node))
        came_from = {}
        g_score = {start_node: 0}
        blocked_cache = {}
        iterations = 0
        
        while open_set:
            iterations += 1
            if iterations > 2000: break
                
            _, current = heapq.heappop(open_set)
            if current == goal_node:
                path = []
                while current in came_from:
                    path.append((current[0]*node_size + node_size/2, current[1]*node_size + node_size/2))
                    current = came_from[current]
                path.reverse()
                path.append(self.goal_pos)
                self.waypoints = path
                return
                
            for dx, dy in [(0,1),(1,0),(0,-1),(-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if neighbor not in blocked_cache:
                    blocked_cache[neighbor] = is_blocked(neighbor[0], neighbor[1])
                if blocked_cache[neighbor]: continue
                
                cost = 1.414 if dx != 0 and dy != 0 else 1.0
                tentative_g = g_score[current] + cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h = math.hypot(goal_node[0]-neighbor[0], goal_node[1]-neighbor[1])
                    heapq.heappush(open_set, (tentative_g + h, neighbor))
                    
        self.waypoints = [self.goal_pos]

    def update(self, keys, obstacles): 
        if self.ai_enabled:
            if any(keys[k] for k in [pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d, 
                                     pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]):
                self.ai_enabled = False

        if self.ai_enabled:
            # 1. 인력 벡터 (A* 웨이포인트 추종)
            if self.waypoints and len(self.waypoints) > 0:
                target = self.waypoints[0]
                dist_to_target = math.hypot(target[0] - self.x, target[1] - self.y)
                if dist_to_target < 120 and len(self.waypoints) > 1:
                    self.waypoints.pop(0)
                    target = self.waypoints[0]
            else:
                target = self.goal_pos

            desired_dx = target[0] - self.x
            desired_dy = target[1] - self.y
            dist_to_t = math.hypot(desired_dx, desired_dy)
            if dist_to_t > 0:
                desired_dx /= dist_to_t
                desired_dy /= dist_to_t
            
            # 길을 따라가려는 힘(인력)
            attraction_force = 2.5 
            final_dx = desired_dx * attraction_force
            final_dy = desired_dy * attraction_force
            
            repulsive_x = 0.0
            repulsive_y = 0.0

            # --- [A. 실제 장애물 선분 - 강한 척력] ---
            sensor_range = 100.0  
            for obs in obstacles:
                for i in range(len(obs) - 1):
                    p1, p2 = obs[i], obs[i+1]
                    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                    if dx == 0 and dy == 0:
                        cx, cy = p1
                    else:
                        t = max(0, min(1, ((self.x - p1[0]) * dx + (self.y - p1[1]) * dy) / (dx*dx + dy*dy)))
                        cx, cy = p1[0] + t * dx, p1[1] + t * dy
                    dist = math.hypot(self.x - cx, self.y - cy)
                    if dist < sensor_range and dist > 0:
                        force_magnitude = ((sensor_range - dist) / sensor_range) ** 2 * 1200.0
                        push_x = (self.x - cx) / dist
                        push_y = (self.y - cy) / dist
                        repulsive_x += push_x * force_magnitude
                        repulsive_y += push_y * force_magnitude

            # --- [B. 보라색 가상 원형 장애물 - 약한 척력] ---
            for obs in obstacles:
                (gx, gy), r = get_virtual_circle(obs)
                dist_to_g = math.hypot(self.x - gx, self.y - gy)
                
                virtual_sensor_range = r + 50.0
                
                if dist_to_g < virtual_sensor_range and dist_to_g > 0:
                    v_force_magnitude = ((virtual_sensor_range - dist_to_g) / virtual_sensor_range) ** 2 * 180.0
                    push_vx = (self.x - gx) / dist_to_g
                    push_vy = (self.y - gy) / dist_to_g
                    repulsive_x += push_vx * v_force_magnitude
                    repulsive_y += push_vy * v_force_magnitude

            # 3. 인력과 이중 척력 합성
            final_dx += repulsive_x
            final_dy += repulsive_y

            target_angle = math.degrees(math.atan2(-final_dy, final_dx)) - 90
            angle_diff = (target_angle - self.angle + 180) % 360 - 180
            
            # 스티어링 세팅
            steering_speed = angle_diff * 0.3
            steering_speed = max(-8, min(8, steering_speed))
            self.angle += steering_speed

            # 4. 자율주행 속도 제어
            obstacle_ahead = self.check_ray_collision(obstacles)
            current_kmh = self.speed * 10
            
            if obstacle_ahead or math.hypot(repulsive_x, repulsive_y) > 400:
                if self.speed > 0: self.speed -= self.acceleration * 1.5 
            else:
                target_spd = self.target_val if abs(angle_diff) < 30 else self.target_val * 0.5
                if current_kmh < target_spd: self.speed += self.acceleration
                elif current_kmh > target_spd + 2: self.speed -= self.acceleration

        else:
            if keys[pygame.K_w]: self.speed += self.acceleration
            if keys[pygame.K_s]: self.speed -= self.acceleration
            if keys[pygame.K_a]: self.angle += 4
            if keys[pygame.K_d]: self.angle -= 4

        real_friction = self.friction_val * 0.005
        internal_max = self.max_speed_limit / 10.0
        if self.speed > 0: self.speed -= real_friction
        elif self.speed < 0: self.speed += real_friction
        if abs(self.speed) < 0.1: self.speed = 0
        if self.speed > internal_max: self.speed = internal_max
        
        rad = math.radians(self.angle + 90)
        self.x += math.cos(rad) * self.speed
        self.y -= math.sin(rad) * self.speed

    def check_collision(self, obstacles):
        rad = math.radians(self.angle + 90)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        hw, hh = self.width / 2, self.height / 2
        corners = [(self.x + (hw*sin_a + hh*cos_a), self.y + (hw*cos_a - hh*sin_a)),
                   (self.x + (-hw*sin_a + hh*cos_a), self.y + (-hw*cos_a - hh*sin_a)),
                   (self.x + (hw*sin_a - hh*cos_a), self.y + (hw*cos_a + hh*sin_a)),
                   (self.x + (-hw*sin_a - hh*cos_a), self.y + (-hw*cos_a + hh*sin_a))]
        
        for obs in obstacles:
            for i in range(len(obs) - 1):
                if dist_to_line_points(corners, obs[i], obs[i+1]):
                    impact_speed = abs(self.speed)
                    if impact_speed > 0.1:
                        self.hp -= (impact_speed ** 2) * 1.2 + 1.0
                        if self.hp < 0: self.hp = 0
                    
                    self.speed *= -0.6
                    self.x += math.cos(rad) * self.speed * 2
                    self.y -= math.sin(rad) * self.speed * 2
                    return True
        return False

    def draw(self, surface, zoom):
        if self.ai_enabled and getattr(self, 'waypoints', None):
            pts = [((p[0]-self.x)*zoom + WIDTH//2, (p[1]-self.y)*zoom + HEIGHT//2) for p in self.waypoints]
            pts.insert(0, (WIDTH//2, HEIGHT//2))
            if len(pts) > 1:
                pygame.draw.lines(surface, PATH_COLOR, False, pts, max(2, int(4*zoom)))

        z_w, z_h = int(self.width * zoom), int(self.height * zoom)
        scaled_img = pygame.transform.scale(self.original_img, (z_w, z_h))
        rotated_img = pygame.transform.rotate(scaled_img, self.angle)
        rect = rotated_img.get_rect(center=(WIDTH//2, HEIGHT//2))
        surface.blit(rotated_img, rect)

    def check_ray_collision(self, obstacles):
        rad = math.radians(self.angle + 90)
        lookahead = 150 
        p0 = (self.x, self.y) 
        p1 = (self.x + math.cos(rad) * lookahead, self.y - math.sin(rad) * lookahead) 
        
        for obs in obstacles:
            for i in range(len(obs) - 1):
                p2, p3 = obs[i], obs[i+1]
                s1_x, s1_y = p1[0] - p0[0], p1[1] - p0[1]
                s2_x, s2_y = p3[0] - p2[0], p3[1] - p2[1]
                
                denom = -s2_x * s1_y + s1_x * s2_y
                if denom != 0:
                    s = (-s1_y * (p0[0] - p2[0]) + s1_x * (p0[1] - p2[1])) / denom
                    t = ( s2_x * (p0[1] - p2[1]) - s2_y * (p0[0] - p2[0])) / denom
                    if 0 <= s <= 1 and 0 <= t <= 1:
                        return True 
        return False

# --- [3. 보조 함수] ---
def dist_to_line_points(corners, p1, p2):
    for c in corners:
        if dist_to_line(c, p1, p2) < 7: return True
    return False

def dist_to_line(p, a, b):
    px, py = p; ax, ay = a; bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0: return math.hypot(px - ax, py - ay)
    t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))

def draw_button(rect, text, mx, my, color=BLUE, t_color=WHITE, font=FONT_S):
    is_hover = rect.collidepoint(mx, my)
    pygame.draw.rect(screen, HOVER_BLUE if is_hover else color, rect, border_radius=5)
    txt_img = font.render(text, True, t_color)
    screen.blit(txt_img, txt_img.get_rect(center=rect.center))
    return is_hover

def draw_hp_bar(hp, max_hp):
    bar_width, bar_height = 250, 30
    x, y = 20, HEIGHT - 50
    pygame.draw.rect(screen, (50, 50, 50), (x, y, bar_width, bar_height), border_radius=5)
    hp_ratio = max(0, hp / max_hp)
    fill_color = (0, 255, 0) if hp_ratio > 0.5 else (255, 165, 0) if hp_ratio > 0.2 else (255, 0, 0)
    pygame.draw.rect(screen, fill_color, (x, y, int(hp_ratio * bar_width), bar_height), border_radius=5)
    screen.blit(FONT_S.render(f"HP: {int(hp)} / {int(max_hp)}", True, WHITE), (x + 10, y + 2))

# --- [4. 메인 루프] ---
def main():
    state = "SETUP"; setup_start = setup_end = None; car = None
    zoom_level = 1.0; show_info = False; game_finished = False
    obstacles = []; drawing_points = []; is_drawing = is_erasing = False

    while True:
        mx, my = pygame.mouse.get_pos()
        events = pygame.event.get()
        if car:
            world_mx = (mx - WIDTH//2) / zoom_level + car.x
            world_my = (my - HEIGHT//2) / zoom_level + car.y

        for event in events:
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            
            if state == "PLAY" and not game_finished:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1: 
                        if show_info:
                            if pygame.Rect(180, 155, 30, 25).collidepoint(mx, my): car.target_val = min(150, car.target_val + 10)
                            elif pygame.Rect(220, 155, 30, 25).collidepoint(mx, my): car.target_val = max(10, car.target_val - 10)
                            elif pygame.Rect(180, 195, 30, 25).collidepoint(mx, my): car.friction_val = min(20, car.friction_val + 1)
                            elif pygame.Rect(220, 195, 30, 25).collidepoint(mx, my): car.friction_val = max(1, car.friction_val - 1)
                            elif not pygame.Rect(10, 140, 260, 110).collidepoint(mx, my): 
                                is_drawing, drawing_points = True, [(world_mx, world_my)]
                        else:
                            is_drawing, drawing_points = True, [(world_mx, world_my)]
                    
                    if event.button == 3: is_erasing = True
                    if event.button == 4: zoom_level = min(zoom_level + 0.1, 1.0)
                    if event.button == 5: zoom_level = max(zoom_level - 0.1, 0.1)

                if event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        is_drawing = False
                        if len(drawing_points) > 1: 
                            obstacles.append(drawing_points)
                            if car: car.recalculate_path(obstacles)
                        drawing_points = []
                    if event.button == 3: is_erasing = False
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_i: show_info = not show_info
                    if event.key == pygame.K_o and car: car.ai_enabled = not car.ai_enabled 

            if event.type == pygame.KEYDOWN and event.key == pygame.K_r and state == "SETUP":
                setup_start = setup_end = None

        if state == "MENU":
            screen.fill(WHITE)
            img = FONT_L.render("AUTO DRIVE", True, BLACK)
            screen.blit(img, img.get_rect(center=(WIDTH//2, 200)))
            if draw_button(pygame.Rect(WIDTH//2-100, 350, 200, 50), "게임 시작", mx, my):
                for e in events:
                    if e.type == pygame.MOUSEBUTTONDOWN: state, obstacles = "SETUP", []

        elif state == "SETUP":
            screen.fill((30, 40, 30))
            for i in range(0, WIDTH, 40): pygame.draw.line(screen, (45, 55, 45), (i, 0), (i, HEIGHT))
            for i in range(0, HEIGHT, 40): pygame.draw.line(screen, (45, 55, 45), (0, i), (WIDTH, i))
            
            if setup_start: pygame.draw.circle(screen, RED, setup_start, 6)
            if setup_end: pygame.draw.circle(screen, BLUE, setup_end, 6)
            if draw_button(pygame.Rect(WIDTH-150, 20, 130, 40), "운전하기", mx, my, color=BLUE if (setup_start and setup_end) else GRAY):
                for e in events:
                    if e.type == pygame.MOUSEBUTTONDOWN and (setup_start and setup_end):
                        state, game_finished = "PLAY", False
                        real_start = (setup_start[0]*WORLD_SCALE, setup_start[1]*WORLD_SCALE)
                        real_end = (setup_end[0]*WORLD_SCALE, setup_end[1]*WORLD_SCALE)
                        car = Car(real_start[0], real_start[1], real_end)
                        dx, dy = real_end[0]-real_start[0], real_end[1]-real_start[1]
                        car.angle = math.degrees(math.atan2(-dy, dx)) - 90
                        car.recalculate_path(obstacles)
            else:
                for e in events:
                    if e.type == pygame.MOUSEBUTTONDOWN:
                        if e.button == 1: setup_start = e.pos
                        elif e.button == 3: setup_end = e.pos

        elif state == "PLAY":
            screen.fill(GREEN)
            if is_erasing:
                eraser_r = 20 / zoom_level
                old_len = len(obstacles)
                obstacles = [o for o in obstacles if not any(dist_to_line((world_mx, world_my), o[i], o[i+1]) < eraser_r for i in range(len(o)-1))]
                if len(obstacles) != old_len and car:
                    car.recalculate_path(obstacles)
            
            if is_drawing: drawing_points.append((world_mx, world_my))
            
            grid_size = int(400 * zoom_level)
            start_x = int(((0 - (car.x * zoom_level - WIDTH//2)) % grid_size)) - grid_size
            start_y = int(((0 - (car.y * zoom_level - HEIGHT//2)) % grid_size)) - grid_size
            for x in range(start_x, WIDTH + grid_size, grid_size): pygame.draw.line(screen, (50, 150, 50), (x, 0), (x, HEIGHT), 1)
            for y in range(start_y, HEIGHT + grid_size, grid_size): pygame.draw.line(screen, (50, 150, 50), (0, y), (WIDTH, y), 1)
            
            goal_sx = (real_end[0] - car.x) * zoom_level + WIDTH//2
            goal_sy = (real_end[1] - car.y) * zoom_level + HEIGHT//2
            pygame.draw.circle(screen, BLUE, (int(goal_sx), int(goal_sy)), int(30 * zoom_level))
            
            # --- [장애물 그리기 (보라색 원과 노란색 벽)] ---
            for obs in obstacles:
                pts = [((p[0]-car.x)*zoom_level+WIDTH//2, (p[1]-car.y)*zoom_level+HEIGHT//2) for p in obs]
                pygame.draw.lines(screen, YELLOW, False, pts, max(2, int(10*zoom_level)))

                (gx, gy), r = get_virtual_circle(obs)
                screen_gx = (gx - car.x) * zoom_level + WIDTH // 2
                screen_gy = (gy - car.y) * zoom_level + HEIGHT // 2
                screen_r = int(r * zoom_level)
                if screen_r > 0:
                    pygame.draw.circle(screen, PURPLE, (int(screen_gx), int(screen_gy)), screen_r, max(1, int(2 * zoom_level)))

            if is_drawing and len(drawing_points) > 1:
                pts = [((p[0]-car.x)*zoom_level+WIDTH//2, (p[1]-car.y)*zoom_level+HEIGHT//2) for p in drawing_points]
                pygame.draw.lines(screen, GRAY, False, pts, max(2, int(10*zoom_level)))

            if not game_finished: 
                car.update(pygame.key.get_pressed(), obstacles) 
                car.check_collision(obstacles)
            car.draw(screen, zoom_level)

            pygame.draw.rect(screen, BLACK, (10, 10, 250, 120), border_radius=10)
            screen.blit(FONT_S.render(f"Speed: {car.speed*10:.0f} km/h", True, WHITE), (25, 20))
            screen.blit(FONT_S.render(f"Dist: {math.hypot(car.x-real_end[0], car.y-real_end[1])/10:.1f} m", True, GRAY), (25, 45))
            screen.blit(FONT_S.render(f"Zoom: {zoom_level:.1f}x", True, HOVER_BLUE), (25, 70))
            ai_col = GREEN if car.ai_enabled else RED
            screen.blit(FONT_S.render(f"AI: {'ON' if car.ai_enabled else 'OFF'}", True, ai_col), (25, 95))
            draw_hp_bar(car.hp, car.max_hp)

            if show_info:
                pygame.draw.rect(screen, (40, 40, 40, 220), (10, 140, 260, 110), border_radius=10)
                screen.blit(FONT_S.render(f"Target: {car.target_val}", True, YELLOW), (25, 155))
                draw_button(pygame.Rect(180, 155, 30, 25), "+", mx, my, GRAY, BLACK, FONT_XS)
                draw_button(pygame.Rect(220, 155, 30, 25), "-", mx, my, GRAY, BLACK, FONT_XS)
                screen.blit(FONT_S.render(f"Friction: {car.friction_val}", True, WHITE), (25, 195))
                draw_button(pygame.Rect(180, 195, 30, 25), "+", mx, my, GRAY, BLACK, FONT_XS)
                draw_button(pygame.Rect(220, 195, 30, 25), "-", mx, my, GRAY, BLACK, FONT_XS)

            if car.hp <= 0 or math.hypot(car.x-real_end[0], car.y-real_end[1]) < 50:
                game_finished = True
                overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA); overlay.fill((0, 0, 0, 150)); screen.blit(overlay, (0,0))
                pygame.draw.rect(screen, WHITE, (WIDTH//2-200, HEIGHT//2-150, 400, 300), border_radius=20)
                img = FONT_L.render("GAME END", True, BLACK)
                screen.blit(img, img.get_rect(center=(WIDTH//2, HEIGHT//2-60)))
                if draw_button(pygame.Rect(WIDTH//2-70, HEIGHT//2+40, 140, 50), "다시하기", mx, my, BLUE):
                    for e in events:
                        if e.type == pygame.MOUSEBUTTONDOWN: state, setup_start, setup_end = "SETUP", None, None

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
