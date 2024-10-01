import agentpy as ap
import numpy as np
import matplotlib.pyplot as plt
import IPython.display

from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder


class Person(ap.Agent):
    def setup(self):
        self.speed = 1
    def move(self):
        print("Move")


class Bicycle(ap.Agent):
    def setup(self):
        self.speed = 1

    def move(self):
        x, y = self.model.grid.positions[self]
        next_position = (x + self.speed) % self.model.p.size, y
        if self.model.is_road[next_position]:
            self.model.grid.move_to(self, next_position)

class TrafficModel(ap.Model):
    def setup(self):
        self.grid = ap.Grid(self, [self.p.size, self.p.size], track_empty=True)
        print(self.grid)
        # Create Bicycle
        self.agents_Bicycle = ap.AgentList(self, self.p.agents_Bicycle, Bicycle)

        # Create Person
        self.agents_Person = ap.AgentList(self, self.p.agents_Person, Person)

        # สุ่มตำแหน่งคน
        num_person_samples = self.p.agents_Person # จำนวนคนที่ต้องการสุ่ม
        positions_person = [(np.random.randint(0, self.p.size), np.random.randint(0, self.p.size)) for _ in range(num_person_samples)]
        self.grid.add_agents(self.agents_Person, positions_person)


        # Set Road Route
        road_y = self.p.size // 2
        road_positions = [(x, road_y) for x in range(self.p.size)]
        # print(road_positions)

        # Set จุดจักรยานที่เป็นการสุ่มบนถนน Road_Position
        positions_bicycle = self.random.sample(road_positions, k=len(self.agents_Bicycle))
        print("start_positions", positions_bicycle) # เช่น [(19, 10)]
        self.grid.add_agents(self.agents_Bicycle, positions_bicycle)

        self.is_road = np.zeros((self.p.size, self.p.size), dtype=bool)
        self.is_road[:, road_y] = True # เลือกคอลัมน์ที่เป็นแนวตั้งทั้งหมด (:) สำหรับแถวที่เป็น road_y ซึ่งเป็นตำแหน่งของถนนที่คุณต้องการให้จักรยานเคลื่อนที่
        # จะได้ is_road ประมาณนี้ array([[False, False, False, False, True, False, False, False]]) กรณีsize 8*8

        self.is_station = np.zeros((self.p.size, self.p.size), dtype=bool)
        self.station_locations = [(5, 9)]  # Fix Station Location
        for each_station in self.station_locations:
            self.is_station[each_station] = True  # สถานี


    def step(self):
        for agent in self.agents_Bicycle:
            agent.move()

    def update(self):
        return self.is_road, self.grid.positions, self.is_station, self.station_locations

def animation_plot(model, ax):
    road, positions, _, stations = model.update() # เป็นการเรียกให้โมเดลอัปเดตสถานะใหม่ในแต่ละ time-step โดยคืนค่าออกมาเป็นสองสิ่ง:
    # road: ข้อมูลเกี่ยวกับถนน เช่น แผนที่ถนนในรูปแบบของ array ที่มีค่าเป็น 0 หรือ 1 (binary) เพื่อแสดงส่วนของถนนและพื้นที่อื่น ๆ
    # positions: ตำแหน่งของจักรยานในโมเดลซึ่งเก็บในรูปแบบ dictionary ที่บอกตำแหน่ง (y, x) ของแต่ละจักรยาน --> การเข้าถึงค่าในอาเรย์สองมิติ โดยปกติจะเรียกผ่านการระบุ (row, column) ซึ่งเทียบได้กับ (y, x) นั่นเอง

    # Plot road
    ax.imshow(road, cmap='binary') # ใช้ ax.imshow() ในการวาดแผนที่ถนน (road) ลงบนแกน ax โดยใช้โหมดสี (colormap) แบบ binary เพื่อแสดงถนนเป็นโทนสีขาวดำ (0 เป็นสีหนึ่งและ 1 เป็นอีกสีหนึ่ง)

    # xticks คือเส้นขีดเล็ก ๆ ที่ปรากฏบนแกน x อะ
    ax.set_xticks(np.arange(-0.5, model.p.size, 1), minor=True)  # ตั้งค่าตำแหน่งเส้นกริดแนวนอน , ตั้งค่าตำแหน่งของ tick marks เพื่อให้แสดงที่ -0.5, 0.5, 1.5, ..., 19.5 ช่วยให้เส้นกริดอยู่กลางระหว่างช่องในกระดานหมากรุก เพื่อให้ผู้ใช้เห็นว่าแต่ละช่องแบ่งออกเป็นอย่างไร
    ax.set_yticks(np.arange(-0.5, model.p.size, 1), minor=True)  # ตั้งค่าตำแหน่งเส้นกริดแนวตั้ง

    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)  # วาดเส้นกริดสีดำ
    ax.tick_params(which='minor', size=0)  # ซ่อน tick-mark

    # แสดงตัวเลขตามขนาด (size)
    ax.set_xticks(np.arange(model.p.size))  # กำหนดตำแหน่งตัวเลขแกน X
    ax.set_yticks(np.arange(model.p.size))  # กำหนดตำแหน่งตัวเลขแกน Y
    ax.set_xticklabels(np.arange(model.p.size))  # กำหนดตัวเลขที่แสดงบนแกน X
    ax.set_yticklabels(np.arange(model.p.size))  # กำหนดตัวเลขที่แสดงบนแกน Y

    # Plot bicycles
    positions_bicycle = {k: v for k, v in model.grid.positions.items() if isinstance(k, Bicycle)}
    # model.grid.positions คือ dictionary ที่เก็บตำแหน่งของตัวแทน (agents) ในโมเดล โดยแต่ละตัวแทน (agent) จะถูกใช้เป็น key และตำแหน่งของมัน (เป็น tuple (x, y)) จะเป็น value 
    # ==> ส่วน items() จะส่งกลับรายการของ key-value pairs ใน dictionary ซึ่งในที่นี้จะเป็นรายการของตัวแทน (agents) และตำแหน่งของพวกเขา.
    # k = แทนตัวแทน (agent) เช่น Bicycle หรือ Person , v = แทนตำแหน่งของตัวแทน (agent) ในรูปแบบ (x, y)
    if positions_bicycle: # ตรวจสอบว่ามีข้อมูลตำแหน่งของจักรยานหรือไม่ หากมีจักรยานใน simulation จะเข้าสู่ขั้นตอนต่อไป
        y, x = zip(*positions_bicycle.values()) # ตำแหน่งของจักรยานที่ได้จาก positions.values() จะถูกแยกออกเป็นพิกัด y และ x ด้วยการใช้ฟังก์ชัน zip() เพื่อเตรียมพิกัดสำหรับการ plot จุด
        ax.scatter(x, y, c='red', s=50) # วาดตำแหน่งจักรยานลงบนแผนที่โดยใช้ ax.scatter() ซึ่งจะ plot จุดที่ตำแหน่ง (x, y) , ใช้สีแดง (c='red') เพื่อแทนจักรยาน และกำหนดขนาดของจุด (s=50)

    # Plot persons
    positions_person = {k: v for k, v in model.grid.positions.items() if isinstance(k, Person)}  # ดึงตำแหน่งของบุคคล
    # positions_person จะเป็น dictionary ใหม่ที่เก็บเฉพาะตำแหน่งของตัวแทนที่เป็นประเภท Person เช่น {Person1: (x1, y1), Person2: (x2, y2),...}
    if positions_person:  # ตรวจสอบว่ามีตำแหน่งของบุคคลหรือไม่
        y_person, x_person = zip(*positions_person.values())  # แยกพิกัด y และ x
        ax.scatter(x_person, y_person, c='blue', s=50)  # วาดตำแหน่งบุคคลด้วยสีฟ้า
    
    y_station, x_station = zip(*stations)  # ใช้ zip เพื่อแยก
    ax.scatter(x_station, y_station, c='green', s=50)  # วาดตำแหน่งบุคคลด้วยสีฟ้า

    ax.set_title(f"Traffic Simulation\n"
                  f"Time-step: {model.t}")




def run_model():
    parameters = {
        'steps': 30,
        'agents_Bicycle': 1, #สร้างจักรยานมา 5
        'agents_Person': 1,
        'size': 20
    }

    fig, ax = plt.subplots(figsize=(7, 7)) #  ขนาดของรูป (figure) ถูกตั้งไว้ที่ 10x10 นิ้ว , fig เป็นตัวแทนของรูปทั้งหมดที่สร้างขึ้น , ax เป็นแกน (axis) ที่จะใช้ในการวาดแผนภาพบนรูปนั้น
    model = TrafficModel(parameters)
    animation = ap.animate(model, fig, ax, animation_plot) # ฟังก์ชัน animate จากไลบรารี AgentPy จะสร้างแอนิเมชันสำหรับการจำลองโมเดล TrafficModel , animation_plot อาจจะเป็นฟังก์ชันที่กำหนดวิธีการแสดงผลการจำลองแต่ละเฟรม

    return IPython.display.HTML(animation.to_jshtml(fps=10))

# Run the model and display the animation
run_model()