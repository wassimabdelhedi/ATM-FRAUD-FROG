"""Visualiseur des bases de données SQLite du projet Fog.

Usage :
  py show_db.py          -> Affiche sous forme de tableau relationnel
  py show_db.py --json   -> Affiche sous forme de JSON formaté
"""
import argparse
import json
import os
import sqlite3

DATABASES = [
    ("fog_a/fog_a_transactions.db", "transactions", "PC 1 : Nœud Fog A (Historique retraits)"),
    ("fog_b/fog_b_locations.db", "card_locations", "PC 2 : Nœud Fog B (Positions des cartes)"),
    ("fog_c/fog_c_decisions.db", "decisions", "PC 3 : Nœud Fog C (Décisions finales)"),
]


def print_table(rows, headers):
    if not rows:
        print("  (Aucune donnée enregistrée)")
        return
    col_widths = [max(len(h), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    fmt = "  " + " | ".join(f"{{:<{w}}}" for w in col_widths)
    sep = "  " + "-+-".join("-" * w for w in col_widths)

    print(fmt.format(*headers))
    print(sep)
    for r in rows:
        print(fmt.format(*[str(x) if x is not None else "NULL" for x in r]))


def inspect(as_json=False):
    for path, table, title in DATABASES:
        print("\n" + "=" * 80)
        print(f"📂 {title}")
        print(f"   Fichier : {path} | Table : {table}")
        print("=" * 80)

        if not os.path.exists(path):
            print("  ⚠️ Ce fichier n'existe pas encore (démarrez le nœud et faites un test).")
            continue

        try:
            conn = sqlite3.connect(path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            rows = cursor.execute(f"SELECT * FROM {table}").fetchall()

            if as_json:
                data = [dict(r) for r in rows]
                print(json.dumps(data, indent=2, ensure_ascii=False))
            else:
                if rows:
                    headers = rows[0].keys()
                    raw_data = [list(r) for r in rows]
                    print_table(raw_data, headers)
                else:
                    print("  (Base vide pour l'instant)")
            conn.close()
        except Exception as e:
            print(f"  Erreur lors de la lecture : {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Afficher en JSON")
    args = parser.parse_args()
    inspect(as_json=args.json)

