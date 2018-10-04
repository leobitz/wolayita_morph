import sqlite3


class CheckpointManager:

    def __init__(self, table_name=None):
        self.table_name = table_name

    def close(self):
        self.conn.commit()
        self.conn.close()

    def open(self):
        self.conn = sqlite3.connect('db.sqlite')
        self.cursor = self.conn.cursor()

    def prepare(self, name):
        self.table_name = name
        q = "CREATE Table IF NOT EXISTS {0} (ID INTEGER PRIMARY KEY AUTOINCREMENT, state text)".format(
            name)
        self.open()
        self.cursor.execute(q)
        self.close()

    def save(self, progress):
        q = "INSERT into {0} (state) values ('{1}')".format(
            self.table_name, progress)
        self.open()
        self.cursor.execute(q)
        self.close()

    def get_last_state(self):
        q = "SELECT * FROM {0} WHERE ID = (SELECT MAX(ID)  FROM {0});".format(
            self.table_name)
        self.open()
        self.cursor.execute(q)
        rows = self.cursor.fetchall()
        self.close()
        if len(rows) >= 1:
            return rows[0]
        else:
            return None

    def get_best_state(self):
        q = "SELECT * FROM {0};".format(
            self.table_name)
        self.open()
        self.cursor.execute(q)
        rows = self.cursor.fetchall()
        best_row = rows[0]
        min_cost = 9999999
        iter = 0
        for row in rows:
            state = row[1]
            vals = state.split(',')
            iter = int(vals[1])
            cost = float(vals[4])
            if cost < min_cost:
                min_cost = cost
                best_row = state
        self.close()
        return best_row, min_cost, iter

    def clear(self):
        q = "DROP TABLE IF EXISTS {0};".format(self.table_name)
        self.open()
        self.cursor.execute(q)
        self.close()

    def get_states(self):
        q = "SELECT * FROM {0};".format(self.table_name)
        self.open()
        self.cursor.execute(q)
        rows = self.cursor.fetchall()
        self.close()
        return rows
        
