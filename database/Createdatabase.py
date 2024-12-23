import sqlite3

def create_database():
    # 連接到 SQLite 資料庫（如果檔案不存在，則會創建一個新的）
    conn = sqlite3.connect('skincare.db')
    cursor = conn.cursor()

    # 創建一個名為 "products" 的資料表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            price REAL NOT NULL,
            type TEXT NOT NULL
        )
    ''')

    # 提交更改並關閉連接
    conn.commit()
    conn.close()

# 呼叫創建資料庫函數
create_database()
def insert_sample_data():
    conn = sqlite3.connect('skincare.db')
    cursor = conn.cursor()

    # 插入一些範例資料
    cursor.execute('''
        INSERT INTO products (type, name, price) VALUES 
        ('dry', 'Curel 潤浸保濕深層乳霜', 1200),
        ('dry', 'Kiehl 契爾氏 高保濕面霜', 1300),
        ('dry', 'La Mer 海藍之謎 修復精華面霜', 7500),
        ('dry', 'Eucerin Aquaphor 高效滋潤修護霜', 450),
        ('dry', 'Neutrogena Hydro Boost 水活保濕凝露', 399),
        ('dry', 'CeraVe PM 夜間保濕乳液', 880),
        ('dry', 'Dr. Jart+ 魚子醬精華修復霜', 1500),
        ('dry', 'LANEIGE 蘭芝 水酷超能修護乳霜', 1200),
        ('dry', 'Aveeno 極潤燕麥保濕乳霜', 350),
        ('dry', 'Bioderma Atoderm Intensive Baume', 1050),
        ('dry', 'Origins 品木宣言 滋潤面霜', 1800),
        ('dry', 'Shiseido 資生堂 未來美肌精華霜', 10500),
        ('dry', 'The Ordinary 100% 冷壓摩洛哥堅果油', 350),
        ('dry', 'La Roche-Posay B5 修復霜', 550),
        ('acne', '理膚寶水 淨痘無瑕極效精華', 950),
        ('acne', '碧兒泉 青春抗痘保濕凝膠', 1500),
        ('acne', '倩碧 溫和潔膚水 4號', 850),
        ('acne', 'DR.WU 杏仁酸溫和煥膚精華液', 800),
        ('acne', 'Bioderma 痘淨化潔膚凝膠', 600),
        ('acne', 'Paula Choice 寶拉珍選 2% 水楊酸精華液', 950),
        ('acne', 'Neutrogena 深層清潔潔面乳', 200),
        ('acne', 'La Roche-Posay 青春潔膚凝膠', 800),
        ('acne', 'The Ordinary 尼克酸胺 10% + 鋅 1% 精華液', 300),
        ('acne', 'Eucerin 抗痘調理面霜', 1200),
        ('acne', 'Innisfree 火山泥毛孔潔膚乳', 590),
        ('acne', 'DR.CINK 玻尿酸控油調理露', 1280),
        ('acne', 'Avène 雅漾 淨柔平衡精華', 900),
        ('acne', 'Kiehl 金盞花植物精萃化妝水', 1200),
        ('acne', 'Cetaphil 溫和保濕潔膚乳', 500),
        ('normal', 'Kiehl 高保濕清爽凝凍', 1300),
        ('normal', 'La Roche-Posay 理膚寶水 溫和保濕乳液', 800),
        ('normal', '倩碧 保濕潤膚霜', 1500),
        ('normal', 'Neutrogena 水活保濕凝露', 399),
        ('normal', 'Bioderma 溫和潔膚乳液', 750),
        ('normal', 'LANEIGE 蘭芝 水酷超能修護霜', 1200),
        ('normal', 'Origins 品木宣言 青春無敵滋潤乳霜', 1800),
        ('normal', 'Eucerin 玻尿酸彈力精華液', 1200),
        ('normal', 'Dr.Wu 玻尿酸保濕精華', 800),
        ('normal', 'Curel 潤浸保濕深層乳液', 1200),
        ('normal', 'Shiseido 資生堂 百優精純乳霜', 2800),
        ('normal', 'Cetaphil 日用保濕乳液 SPF15', 600),
        ('normal', 'Avène 雅漾 保濕舒緩霜', 900),
        ('normal', 'The Ordinary 天然保濕因子乳液', 350),
        ('normal', 'Clarins 娇韵诗 平衡柔膚水', 2200),
        ('oily', '理膚寶水 清痘淨膚雙效精華', 950),
        ('oily', '碧兒泉 清脂抗痘保濕凝膠', 1500),
        ('oily', '倩碧 男士控油無油凝膠', 1200),
        ('oily', 'Kiehl 契爾氏 高效清爽無油保濕凝露', 1300),
        ('oily', 'Innisfree 綠茶無油保濕凝膠', 600),
        ('oily', 'Neutrogena 深層控油潔面乳', 200),
        ('oily', 'Paula Choice 10% 尼克酸胺精華', 1500),
        ('oily', 'La Roche-Posay 青春潔膚凝膠', 800),
        ('oily', 'Eucerin Oil Control 抗痘調理面霜', 1200),
        ('oily', '黛珂 抗痘清爽化妝水', 1500)
    ''')

    conn.commit()
    conn.close()

# 呼叫插入資料函數
insert_sample_data()

