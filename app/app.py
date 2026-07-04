from flask import Flask, render_template, request, jsonify
import sqlite3
from pathlib import Path
import datetime
from dateutil.relativedelta import relativedelta

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / 'data' / 'medicsafc.db'

app = Flask(__name__, static_folder='static', template_folder='templates')


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS pacientes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tipo_paciente TEXT,
                paciente_seleccionado TEXT,
                nombre TEXT,
                apellido TEXT,
                fecha_nacimiento TEXT,
                edad INTEGER,
                sexo TEXT,
                peso REAL,
                talla REAL,
                alergias TEXT,
                antecedentes TEXT,
                telefono TEXT,
                correo TEXT,
                domicilio TEXT,
                colonia TEXT,
                municipio TEXT,
                estado TEXT,
                codigo_postal TEXT,
                notas TEXT,
                creado_en TEXT
            )
        ''')
        conn.commit()


init_db()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/pacientes', methods=['GET'])
def listar_pacientes():
    with get_conn() as conn:
        rows = conn.execute('SELECT * FROM pacientes ORDER BY creado_en DESC').fetchall()
    data = []
    for r in rows:
        data.append({
            'id': r['id'],
            'tipo_paciente': r['tipo_paciente'],
            'paciente_seleccionado': r['paciente_seleccionado'],
            'nombre': r['nombre'],
            'apellido': r['apellido'],
            'fecha_nacimiento': r['fecha_nacimiento'],
            'edad': r['edad'],
            'sexo': r['sexo'],
            'peso': r['peso'],
            'talla': r['talla'],
            'alergias': r['alergias'],
            'antecedentes': r['antecedentes'],
            'telefono': r['telefono'],
            'correo': r['correo'],
            'domicilio': r['domicilio'],
            'colonia': r['colonia'],
            'municipio': r['municipio'],
            'estado': r['estado'],
            'codigo_postal': r['codigo_postal'],
            'notas': r['notas'],
            'creado_en': r['creado_en'],
        })
    return jsonify(data)


@app.route('/api/pacientes', methods=['POST'])
def crear_paciente():
    payload = request.get_json(force=True) or {}
    payload['edad'] = calc_edad(payload.get('fecha_nacimiento'))
    creado_en = datetime.datetime.now().isoformat(timespec='seconds')
    with get_conn() as conn:
        conn.execute('''
            INSERT INTO pacientes (
                tipo_paciente, paciente_seleccionado, nombre, apellido, fecha_nacimiento, edad, sexo,
                peso, talla, alergias, antecedentes, telefono, correo,
                domicilio, colonia, municipio, estado, codigo_postal, notas, creado_en
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ''', (
            payload.get('tipo_paciente'),
            payload.get('paciente_seleccionado'),
            payload.get('nombre'),
            payload.get('apellido'),
            payload.get('fecha_nacimiento'),
            payload.get('edad'),
            payload.get('sexo'),
            payload.get('peso'),
            payload.get('talla'),
            payload.get('alergias'),
            payload.get('antecedentes'),
            payload.get('telefono'),
            payload.get('correo'),
            payload.get('domicilio'),
            payload.get('colonia'),
            payload.get('municipio'),
            payload.get('estado'),
            payload.get('codigo_postal'),
            payload.get('notas'),
            creado_en,
        ))
        conn.commit()
    return jsonify({'status': 'ok', 'creado_en': creado_en, 'id': conn.execute('SELECT last_insert_rowid()').fetchone()[0]})


@app.route('/api/pacientes/<int:paciente_id>', methods=['PUT'])
def actualizar_paciente(paciente_id: int):
    payload = request.get_json(force=True) or {}
    payload['edad'] = calc_edad(payload.get('fecha_nacimiento'))
    with get_conn() as conn:
        conn.execute('''
            UPDATE pacientes SET
                tipo_paciente = COALESCE(?, tipo_paciente),
                paciente_seleccionado = COALESCE(?, paciente_seleccionado),
                nombre = COALESCE(?, nombre),
                apellido = COALESCE(?, apellido),
                fecha_nacimiento = COALESCE(?, fecha_nacimiento),
                edad = COALESCE(?, edad),
                sexo = COALESCE(?, sexo),
                peso = COALESCE(?, peso),
                talla = COALESCE(?, talla),
                alergias = COALESCE(?, alergias),
                antecedentes = COALESCE(?, antecedentes),
                telefono = COALESCE(?, telefono),
                correo = COALESCE(?, correo),
                domicilio = COALESCE(?, domicilio),
                colonia = COALESCE(?, colonia),
                municipio = COALESCE(?, municipio),
                estado = COALESCE(?, estado),
                codigo_postal = COALESCE(?, codigo_postal),
                notas = COALESCE(?, notas)
            WHERE id = ?
        ''', tuple([
            payload.get('tipo_paciente'), payload.get('paciente_seleccionado'),
            payload.get('nombre'), payload.get('apellido'),
            payload.get('fecha_nacimiento'), payload.get('edad'), payload.get('sexo'),
            payload.get('peso'), payload.get('talla'),
            payload.get('alergias'), payload.get('antecedentes'),
            payload.get('telefono'), payload.get('correo'),
            payload.get('domicilio'), payload.get('colonia'),
            payload.get('municipio'), payload.get('estado'), payload.get('codigo_postal'),
            payload.get('notas'), paciente_id
        ]))
        conn.commit()
    return jsonify({'status': 'updated'})


@app.route('/api/pacientes/<int:paciente_id>', methods=['DELETE'])
def eliminar_paciente(paciente_id: int):
    with get_conn() as conn:
        conn.execute('DELETE FROM pacientes WHERE id = ?', (paciente_id,))
        conn.commit()
    return jsonify({'status': 'deleted'})


def calc_edad(fecha_nacimiento):
    if not fecha_nacimiento:
        return None
    try:
        d = datetime.date.fromisoformat(str(fecha_nacimiento))
        now = datetime.date.today()
        edad = relativedelta(now, d).years
        return edad
    except Exception:
        return None


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5050, debug=False)
