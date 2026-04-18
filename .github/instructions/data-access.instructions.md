---
applyTo: "app/data_access/**/*.py"
---

## Adatelérési réteg – szabályok

### DataManager használata
- Soha ne írj nyers SQL string-et a business logic rétegbe
- Mindig a `DataManager` CRUD metódusait használd
- Ha új DB műveletre van szükség, először a `DataManager`-be add hozzá

### Sémamódosítás
- Meglévő tábla/oszlop struktúrát **ne törd el visszafelé**
- Új oszlopot `ALTER TABLE ... ADD COLUMN` + default értékkel adj hozzá
- Sémaváltozáshoz írj migrációs szkriptet `scripts/` alá

### SQLite idiomatikus minták
```python
# ✅ Helyes – paraméteres query (SQL injection védelem)
cursor.execute("SELECT * FROM decisions WHERE ticker = ?", (ticker,))

# ❌ Tilos – string interpoláció
cursor.execute(f"SELECT * FROM decisions WHERE ticker = '{ticker}'")
```

### Tranzakciókezelés
- Hosszabb write műveleteknél használj `with conn:` context manager-t
- Rollback automatikus exceptionkor a context manager esetén
