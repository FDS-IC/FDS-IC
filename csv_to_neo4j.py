from py2neo import Graph, Node, Relationship
import csv
import re

# 连接 Neo4j 数据库
graph = Graph("bolt://localhost:7687", auth=("neo4j", "123456ln"))
graph.delete_all()  # 清空数据库

# 可选：清空已有图数据（小心操作）
# graph.delete_all()

# === 2. 中文标点清洗函数 ===
def clean_text(text):
    if not text:
        return ""
    # 替换常见中文标点为英文标点
    text = re.sub(r"[，]", ",", text)
    text = re.sub(r"[。]", ".", text)
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"[：]", ":", text)
    text = re.sub(r"[；]", ";", text)
    text = re.sub(r"\s+", "", text)  # 去除空格换行
    return text.strip()

# === 3. 读取 CSV 文件 ===
file_path = "D:/pythonProject/BERT-BILSTM-CRF-main/output/data_1.csv"
with open(file_path, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) < 7:
            continue  # 跳过不完整行

        # === 4. 字段提取与清洗 ===
        trigger = clean_text(row[0])
        event = clean_text(row[1])
        time = clean_text(row[2])
        location = clean_text(row[3])
        cause = clean_text(row[4])
        loss = clean_text(row[5])
        measure = clean_text(row[6])

        # === 5. 创建节点 ===
        trigger_node = Node("Trigger", name=trigger)
        event_node = Node("事件", name=event)
        time_node = Node("时间", name=time)
        location_node = Node("地点", name=location)
        cause_node = Node("原因", name=cause)
        loss_node = Node("伤亡", name=loss)
        measure_node = Node("处理措施", name=measure)

        # === 6. 去重创建（确保唯一）===
        graph.merge(trigger_node, "Trigger", "name")
        graph.merge(event_node, "事件", "name")
        graph.merge(time_node, "时间", "name")
        graph.merge(location_node, "地点", "name")
        graph.merge(cause_node, "原因", "name")
        graph.merge(loss_node, "伤亡", "name")
        graph.merge(measure_node, "处理措施", "name")

        # === 7. 创建关系（触发词为中心）===
        graph.merge(Relationship(trigger_node, "属于事件", event_node))
        graph.merge(Relationship(trigger_node, "发生时间", time_node))
        graph.merge(Relationship(trigger_node, "发生地点", location_node))
        graph.merge(Relationship(trigger_node, "事故原因", cause_node))
        graph.merge(Relationship(trigger_node, "造成伤亡", loss_node))
        graph.merge(Relationship(trigger_node, "应急处理", measure_node))
