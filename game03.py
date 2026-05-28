import pygame
import math
import sys
import heapq 

# --- [1. 설정 값] ---
pygame.init()
WIDTH, HEIGHT = 1000, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Auto Drive - Dual A* (Global & Local Escape)")
clock = pygame.time.Clock()

WORLD_SCALE = 10
WHITE, BLACK, GRAY = (255, 255, 255), (0, 0, 0), (200, 200, 200)
GREEN, RED, BLUE, YELLOW = (34, 139, 34), (255, 0, 0), (0, 120, 215), (255, 220, 0)
HOVER_BLUE = (0, 180, 255)
ORANGE = (255, 165, 0) # HEAVY 모드 탈출용 지역 A* 경로 색상
PATH_COLOR = (255, 105, 180) # 전역 A* 경로 색상
FONT_L = pygame.font.SysFont("malgungothic", 50, bold=True)
FONT_S = pygame.font.SysFont("malgungothic", 20)
FONT_XS = pygame.font.SysFont("malgungothic", 16, bold=True)

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
        
        self.ai_state = "OFF"      
        
        # [핵심] 이중 A* 경로 분리
        self.global_waypoints = [] # 목적지까지의 전역 경로
        self.local_waypoints = []  # 장애물 탈출용 지역 경로 (HEAVY 전용)
        
        self.current_target = None
        self.wp_start_time = 0
        self.wp_expected_time = 0
        
        self.max_hp = 500.0
        self.hp = self.max_hp
        
        self.width, self.height = 40, 70
        self.original_img = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.rect(self.original_img, RED, (0, 0, self.width, self.height), border_radius=8)
        pygame.draw.rect(self.original_img, BLACK, (5, 10, self.width-10, 20)) 

    # --- [재사용 가능한 A* 길찾기 함수] ---
    def get_astar_path(self, start_pos, target_pos, obstacles, node_size):
        start_node = (int(start_pos[0] // node_size), int(start_pos[1] // node_size))
        goal_node = (int(target_pos[0] // node_size), int(target_pos[1] // node_size))

        def is_blocked(nx, ny):
            cx = nx * node_size + node_size / 2
            cy = ny * node_size + node_size / 2
            
            # 탐색 시작점 주변 1.5칸 반경은 무조건 뚫려있다고 판단 (벽에 박혀있을 때 길찾기 포기 방지)
            if math.hypot(cx - start_pos[0], cy - start_pos[1]) < node_size * 1.5:
                return False
                
            for obs in obstacles:
                for i in range(len(obs)-1):
                    # 차폭 고려 충돌 검사
                    if dist_to_line((cx, cy), obs[i], obs[i+1]) < 35: 
                        return True
            return False

        open_set = []
        heapq.heappush(open_set, (0, start_node))
        came_from = {}
        g_score = {start_node: 0}
        blocked_cache = {}
        iterations = 0
        
        while open_set:
            iterations += 1
            if iterations > 3000: break # 과부하 방지
                
            _, current = heapq.heappop(open_set)
            
            # 목표에 도달했다면 경로 재구성
            if current == goal_node:
                path = []
                while current in came_from:
                    path.append((current[0]*node_size + node_size/2, current[1]*node_size + node_size/2))
                    current = came_from[current]
                path.reverse()
                path.append(target_pos)
                
                # 시작 위치와 겹치는 첫 번째 불필요한 노드는 삭제
                while len(path) > 1 and math.hypot(path[0][0] - start_pos[0], path[0][1] - start_pos[1]) < node_size * 0.8:
                    path.pop(0)
                return path
                
            # 8방향 탐색
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
                    
        # 길을 찾지 못했다면 일단 목표를 향해 직선 반환 (포기 방지)
        return [target_pos]

    # [전역 경로 재탐색 호출]
    def recalculate_global_path(self, obstacles):
        # 전역 맵핑은 80px 단위로 큼직하고 빠르게 계산
        self.global_waypoints = self.get_astar_path((self.x, self.y), self.goal_pos, obstacles, node_size=80)
        self.ai_state = "LIGHT" # 장애물이 갱신되면 다시 LIGHT 모드로 초기화

    def update(self, keys, obstacles): 
        # 수동 개입 시 AI OFF
        if self.ai_state != "OFF":
            if any(keys[k] for k in [pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d, 
                                     pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]):
                self.ai_state = "OFF"

        current_time = pygame.time.get_ticks()

        if self.ai_state != "OFF":
            
            # ==============================================================
            # [STATE 1: LIGHT-DRIVING] - 전역 경로(Global) + APF 스티어링
            # ==============================================================
            if self.ai_state == "LIGHT":
                if self.global_waypoints and len(self.global_waypoints) > 0:
                    target = self.global_waypoints[0]
                    dist_to_target = math.hypot(target[0] - self.x, target[1] - self.y)
                    
                    if self.current_target != target:
                        self.current_target = target
                        self.wp_start_time = current_time
                        self.wp_expected_time = (dist_to_target / 100.0) * 1000

                    # 전역 웨이포인트 도달 판정
                    if dist_to_target < 80 and len(self.global_waypoints) > 1:
                        self.global_waypoints.pop(0)
                        target = self.global_waypoints[0]
                        self.current_target = target
                        self.wp_start_time = current_time
                        self.wp_expected_time = (math.hypot(target[0] - self.x, target[1] - self.y) / 100.0) * 1000

                    # [핵심] 갇힘 판단 타이머 -> 지역 A* 가동!
                    elapsed_time = current_time - self.wp_start_time
                    if elapsed_time > self.wp_expected_time + 1500:
                        self.ai_state = "HEAVY"
                        # 현재 갇힌 내 위치에서 -> 내가 원래 가려던 다음 전역 웨이포인트까지만
                        # 40px 단위로 아주 촘촘한 탈출용 지역 A*를 돌립니다!
                        self.local_waypoints = self.get_astar_path((self.x, self.y), target, obstacles, node_size=40)
                        
                else:
                    target = self.goal_pos

                # --- APF 조향 연산 ---
                desired_dx = target[0] - self.x
                desired_dy = target[1] - self.y
                dist_to_t = math.hypot(desired_dx, desired_dy)
                if dist_to_t > 0:
                    desired_dx /= dist_to_t
                    desired_dy /= dist_to_t

                attraction_force = 2.5 
                final_dx = desired_dx * attraction_force
                final_dy = desired_dy * attraction_force
                
                repulsive_x, repulsive_y = 0.0, 0.0
                sensor_range = 100.0  
                for obs in obstacles:
                    for i in range(len(obs) - 1):
                        p1, p2 = obs[i], obs[i+1]
                        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                        if dx == 0 and dy == 0: cx, cy = p1
                        else:
                            t = max(0, min(1, ((self.x - p1[0]) * dx + (self.y - p1[1]) * dy) / (dx*dx + dy*dy)))
                            cx, cy = p1[0] + t * dx, p1[1] + t * dy
                        dist = math.hypot(self.x - cx, self.y - cy)
                        if dist < sensor_range and dist > 0:
                            force_magnitude = ((sensor_range - dist) / sensor_range) ** 2 * 1200.0
                            repulsive_x += ((self.x - cx) / dist) * force_magnitude
                            repulsive_y += ((self.y - cy) / dist) * force_magnitude

                final_dx += repulsive_x
                final_dy += repulsive_y

                target_angle = math.degrees(math.atan2(-final_dy, final_dx)) - 90
                angle_diff = (target_angle - self.angle + 180) % 360 - 180
                
                steering_speed = max(-8, min(8, angle_diff * 0.3))
                self.angle += steering_speed

                obstacle_ahead = self.check_ray_collision(obstacles)
                current_kmh = self.speed * 10
                
                if obstacle_ahead or math.hypot(repulsive_x, repulsive_y) > 400:
                    if self.speed > 0: self.speed -= self.acceleration * 1.5 
                else:
                    target_spd = self.target_val if abs(angle_diff) < 30 else self.target_val * 0.5
                    if current_kmh < target_spd: self.speed += self.acceleration
                    elif current_kmh > target_spd + 2: self.speed -= self.acceleration

            # ==============================================================
            # [STATE 2: HEAVY-DRIVING] - 지역 경로(Local) 맹목적 추종
            # ==============================================================
            elif self.ai_state == "HEAVY":
                if self.local_waypoints and len(self.local_waypoints) > 0:
                    local_target = self.local_waypoints[0]
                    dist_to_local = math.hypot(local_target[0] - self.x, local_target[1] - self.y)
                    
                    # 40px 단위 맵핑이므로 도달 판정 거리도 짧게(35px) 설정
                    if dist_to_local < 35:
                        self.local_waypoints.pop(0)
                        
                        # 지역 경로를 끝까지 다 따라갔다면 = 탈출 완료!
                        if len(self.local_waypoints) == 0:
                            self.ai_state = "LIGHT"
                            
                            # 탈출 지점이 곧 전역 웨이포인트였으므로, 전역 웨이포인트도 하나 지워줌
                            if self.global_waypoints:
                                self.global_waypoints.pop(0)
                                
                            # 타이머 초기화 세팅 후 리턴 (다음 프레임부터 LIGHT 모드로 다시 시작)
                            if self.global_waypoints:
                                target = self.global_waypoints[0]
                                self.current_target = target
                                self.wp_start_time = current_time
                                self.wp_expected_time = (math.hypot(target[0] - self.x, target[1] - self.y) / 100.0) * 1000
                            return 
                        else:
                            local_target = self.local_waypoints[0]
                else:
                    self.ai_state = "LIGHT"
                    return
                
                # 척력을 무시하고 '지역 타겟(주황색 선)'을 향해 다이렉트로 몸을 돌림
                desired_dx = local_target[0] - self.x
                desired_dy = local_target[1] - self.y
                target_angle = math.degrees(math.atan2(-desired_dy, desired_dx)) - 90
                angle_diff = (target_angle - self.angle + 180) % 360 - 180
                
                steering_speed = angle_diff * 0.6
                steering_speed = max(-15, min(15, steering_speed))
                self.angle += steering_speed
                
                # 탈출용 피벗 턴 (제자리 회전)
                if abs(angle_diff) > 25:
                    target_spd = 15 # 각도가 틀어져 있으면 천천히 차체부터 회전
                else:
                    target_spd = 50 # 출구를 바라보게 되면 과감하게 전진!
                    
                current_kmh = self.speed * 10
                if current_kmh < target_spd: self.speed += self.acceleration * 1.5
                elif current_kmh > target_spd + 2: self.speed -= self.acceleration * 2

        # ==============================================================
        # [STATE 0: AI OFF] - 수동 조작
        # ==============================================================
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

    def check_ray_collision(self, obstacles, lookahead=150):
        rad = math.radians(self.angle + 90)
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

    def draw(self, surface, zoom):
        if self.ai_state != "OFF":
            # 1. 전역 경로 그리기 (분홍색 선)
            if getattr(self, 'global_waypoints', None):
                pts = [((p[0]-self.x)*zoom + WIDTH//2, (p[1]-self.y)*zoom + HEIGHT//2) for p in self.global_waypoints]
                pts.insert(0, (WIDTH//2, HEIGHT//2))
                if len(pts) > 1:
                    pygame.draw.lines(surface, PATH_COLOR, False, pts, max(2, int(4*zoom)))
            
            # 2. 지역 경로 그리기 (HEAVY 모드일 때만 보이는 주황색 탈출 경로!)
            if self.ai_state == "HEAVY" and getattr(self, 'local_waypoints', None):
                pts = [((p[0]-self.x)*zoom + WIDTH//2, (p[1]-self.y)*zoom + HEIGHT//2) for p in self.local_waypoints]
                pts.insert(0, (WIDTH//2, HEIGHT//2))
                if len(pts) > 1:
                    pygame.draw.lines(surface, ORANGE, False, pts, max(2, int(6*zoom))) # 탈출선은 두껍게

        z_w, z_h = int(self.width * zoom), int(self.height * zoom)
        scaled_img = pygame.transform.scale(self.original_img, (z_w, z_h))
        rotated_img = pygame.transform.rotate(scaled_img, self.angle)
        rect = rotated_img.get_rect(center=(WIDTH//2, HEIGHT//2))
        surface.blit(rotated_img, rect)

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
                            if car: car.recalculate_global_path(obstacles) # 장애물 갱신 시 전역 경로 재계산
                        drawing_points = []
                    if event.button == 3: is_erasing = False
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_i: show_info = not show_info
                    if event.key == pygame.K_o and car: 
                        car.ai_state = "LIGHT" if car.ai_state == "OFF" else "OFF"
                        car.current_target = None 

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
                        car.recalculate_global_path(obstacles) # 시작 시 전역 경로 계산
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
                    car.recalculate_global_path(obstacles) # 장애물 지울 때도 전역 경로 갱신
            
            if is_drawing: drawing_points.append((world_mx, world_my))
            
            grid_size = int(400 * zoom_level)
            start_x = int(((0 - (car.x * zoom_level - WIDTH//2)) % grid_size)) - grid_size
            start_y = int(((0 - (car.y * zoom_level - HEIGHT//2)) % grid_size)) - grid_size
            for x in range(start_x, WIDTH + grid_size, grid_size): pygame.draw.line(screen, (50, 150, 50), (x, 0), (x, HEIGHT), 1)
            for y in range(start_y, HEIGHT + grid_size, grid_size): pygame.draw.line(screen, (50, 150, 50), (0, y), (WIDTH, y), 1)
            
            goal_sx = (real_end[0] - car.x) * zoom_level + WIDTH//2
            goal_sy = (real_end[1] - car.y) * zoom_level + HEIGHT//2
            pygame.draw.circle(screen, BLUE, (int(goal_sx), int(goal_sy)), int(30 * zoom_level))
            
            for obs in obstacles:
                pts = [((p[0]-car.x)*zoom_level+WIDTH//2, (p[1]-car.y)*zoom_level+HEIGHT//2) for p in obs]
                pygame.draw.lines(screen, YELLOW, False, pts, max(2, int(10*zoom_level)))

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
            
            if car.ai_state == "LIGHT": ai_col = GREEN
            elif car.ai_state == "HEAVY": ai_col = ORANGE
            else: ai_col = RED
            screen.blit(FONT_S.render(f"AI: {car.ai_state}", True, ai_col), (25, 95))
            
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
