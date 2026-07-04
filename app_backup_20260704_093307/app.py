from flask import Flask, render_template, request, jsonify
import sqlite3
from pathlib import Path
import datetime

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
                nombre_completo TEXT,
                fecha_nacimiento TEXT,
                url_chat_detectado TEXT,
                nombre_paciente_detectado TEXT,
                paciente_id TEXT,
                dia_nacimiento TEXT,
                mes_nacimiento TEXT,
                anio_nacimiento TEXT,
                correo TEXT,
                telefono TEXT,
                sexo TEXT,
                peso REAL,
                talla REAL,
                alergias TEXT,
                antecedentes TEXT,
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
            'nombre_completo': r['nombre_completo'],
            'fecha_nacimiento': r['fecha_nacimiento'],
            'url_chat_detectado': r['url_chat_detectado'],
            'nombre_paciente_detectado': r['nombre_paciente_detectado'],
            'paciente_id': r['paciente_id'],
            'dia_nacimiento': r['dia_nacimiento'],
            'mes_nacimiento': r['mes_nacimiento'],
            'anio_nacimiento': r['anio_nacimiento'],
            'correo': r['correo'],
            'telefono': r['telefono'],
            'sexo': r['sexo'],
            'peso': r['peso'],
            'talla': r['talla'],
            'alergias': r['alergias'],
            'antecedentes': r['antecedentes'],
            'notas': r['notas'],
            'creado_en': r['creado_en'],
        })
    return jsonify(data)


@app.route('/api/pacientes', methods=['POST'])
def crear_paciente():
    payload = request.get_json(force=True) or {}
    creado_en = datetime.datetime.now().isoformat(timespec='seconds')
    with get_conn() as conn:
        conn.execute('''
            INSERT INTO pacientes (
                tipo_paciente, paciente_seleccionado, nombre_completo, fecha_nacimiento,
                url_chat_detectado, nombre_paciente_detectado, paciente_id,
                dia_nacimiento, mes_nacimiento, anio_nacimiento,
                correo, telefono, sexo, peso, talla, alergias, antecedentes, notas, creado_en
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ''', (
            payload.get('tipo_paciente'),
            payload.get('paciente_seleccionado'),
            payload.get('nombre_completo'),
            payload.get('fecha_nacimiento'),
            payload.get('url_chat_detectado'),
            payload.get('nombre_paciente_detectado'),
            payload.get('paciente_id'),
            payload.get('dia_nacimiento'),
            payload.get('mes_nacimiento'),
            payload.get('anio_nacimiento'),
            payload.get('correo'),
            payload.get('telefono'),
            payload.get('sexo'),
            payload.get('peso'),
            payload.get('talla'),
            payload.get('alergias'),
            payload.get('antecedentes'),
            payload.get('notas'),
            creado_en,
        ))
        conn.commit()
    return jsonify({'status': 'ok', 'creado_en': creado_en})


@app.route('/api/pacientes/<int:paciente_id>', methods=['PUT'])
def actualizar_paciente(paciente_id: int):
    payload = request.get_json(force=True) or {}
    with get_conn() as conn:
        conn.execute('''
            UPDATE pacientes SET
                tipo_paciente = COALESCE(?, tipo_paciente),
                paciente_seleccionado = COALESCE(?, paciente_seleccionado),
                nombre_completo = COALESCE(?, nombre_completo),
                fecha_nacimiento = COALESCE(?, fecha_nacimiento),
                url_chat_detectado = COALESCE(?, url_chat_detectado),
                nombre_paciente_detectado = COALESCE(?, nombre_paciente_detectado),
                paciente_id = COALESCE(?, paciente_id),
                dia_nacimiento = COALESCE(?, dia_nacimiento),
                mes_nacimiento = COALESCE(?, mes_nacimiento),
                anio_nacimiento = COALESCE(?, anio_nacimiento),
                correo = COALESCE(?, correo),
                telefono = COALESCE(?, telefono),
                sexo = COALESCE(?, sexo),
                peso = COALESCE(?, peso),
                talla = COALESCE(?, talla),
                alergias = COALESCE(?, alergias),
                antecedentes = COALESCE(?, antecedentes),
                notas = COALESCE(?, notas)
            WHERE id = ?
        ''', tuple([
            payload.get(k) for k in [
                'tipo_paciente', 'paciente_seleccionado', 'nombre_completo', 'fecha_nacimiento',
                'url_chat_detectado', 'nombre_paciente_detectado', 'paciente_id',
                'dia_nacimiento', 'mes_nacimiento', 'anio_nacimiento',
                'correo', 'telefono', 'sexo', 'peso', 'talla',
                'alergias', 'antecedentes', 'notas'
            ]] + [paciente_id]))
        conn.commit()
    return jsonify({'status': 'updated'})


@app.route('/api/pacientes/<int:paciente_id>', methods=['DELETE'])
def eliminar_paciente(paciente_id: int):
    with get_conn() as conn:
        conn.execute('DELETE FROM pacientes WHERE id = ?', (paciente_id,))
        conn.commit()
    return jsonify({'status': 'deleted'})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5050, debug=False)
