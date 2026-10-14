"""SQLite catalog for non-destructive culling."""
import sqlite3
from pathlib import Path
EXTENSIONS={'.jpg','.jpeg','.png','.tif','.tiff','.bmp','.webp','.dng','.cr2','.cr3','.nef','.arw','.raf','.orf','.rw2','.nrw','.pef','.3fr','.fff','.iiq','.mos','.mrw','.rwl','.x3f'}
class Catalog:
    def __init__(self,path):
        self.db=sqlite3.connect(str(path))
        self.db.execute('CREATE TABLE IF NOT EXISTS photos (path TEXT PRIMARY KEY, rating INTEGER NOT NULL DEFAULT 0, pick INTEGER NOT NULL DEFAULT 0)')
        self.db.commit()
    def import_folder(self,folder):
        paths=sorted(str(p.resolve()) for p in Path(folder).rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
        self.db.executemany('INSERT OR IGNORE INTO photos(path) VALUES (?)',((p,) for p in paths))
        self.db.commit()
        return len(paths)
    def entries(self,mode='All'):
        clauses={'All':'1=1','Picks':'pick=1','Rejected':'pick=-1','Unflagged':'pick=0','Rated':'rating>0'}
        return self.db.execute('SELECT path,rating,pick FROM photos WHERE '+clauses[mode]+' ORDER BY path').fetchall()
    def rate(self,path,rating):
        if rating not in range(6): raise ValueError('rating must be 0..5')
        self.db.execute('UPDATE photos SET rating=? WHERE path=?',(rating,path));self.db.commit()
    def flag(self,path,pick):
        if pick not in (-1,0,1): raise ValueError('invalid flag')
        self.db.execute('UPDATE photos SET pick=? WHERE path=?',(pick,path));self.db.commit()
