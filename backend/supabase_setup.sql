-- ══════════════════════════════════════════════════════════════
-- Clínica Cobba — Script SQL para Supabase
-- Ejecutar en: Supabase → SQL Editor → New Query
-- ══════════════════════════════════════════════════════════════

-- ── Tabla de pacientes ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS patients (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  first_name  TEXT NOT NULL,
  last_name   TEXT NOT NULL,
  dni         TEXT UNIQUE NOT NULL,
  created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- ── Tabla de citas ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS appointments (
  id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_name TEXT NOT NULL,
  dni          TEXT NOT NULL,
  doctor       TEXT NOT NULL,
  specialty    TEXT NOT NULL,
  date         DATE NOT NULL,
  time         TEXT NOT NULL,
  status       TEXT NOT NULL DEFAULT 'Confirmada'
                CHECK (status IN ('Confirmada', 'Pendiente', 'No-Show', 'Cancelada')),
  created_at   TIMESTAMPTZ DEFAULT NOW(),
  FOREIGN KEY (dni) REFERENCES patients(dni) ON DELETE CASCADE
);

-- ── Datos de ejemplo ──────────────────────────────────────────
INSERT INTO patients (first_name, last_name, dni) VALUES
  ('Carlos',  'Ruiz',    '72345678'),
  ('Ana',     'Gomez',   '45678912'),
  ('Luis',    'Merino',  '12345678'),
  ('Sofia',   'Castro',  '76543210'),
  ('Pedro',   'Huaman',  '87654321'),
  ('Rosa',    'Linares', '11223344')
ON CONFLICT (dni) DO NOTHING;

INSERT INTO appointments (patient_name, dni, doctor, specialty, date, time, status) VALUES
  ('Carlos Ruiz',   '72345678', 'Dr. Quispe',   'Odontología General', '2026-07-07', '09:00', 'Confirmada'),
  ('Ana Gomez',     '45678912', 'Dra. Paz',     'Ortodoncia',          '2026-07-07', '10:30', 'Pendiente'),
  ('Luis Merino',   '12345678', 'Dr. Flores',   'Endodoncia',          '2026-07-08', '08:00', 'No-Show'),
  ('Sofia Castro',  '76543210', 'Dra. Torres',  'Periodoncia',         '2026-07-08', '16:00', 'Confirmada'),
  ('Pedro Huaman',  '87654321', 'Dr. Mendoza',  'Implantología',       '2026-07-09', '08:00', 'Confirmada'),
  ('Rosa Linares',  '11223344', 'Dra. Ríos',    'Odontopediatría',     '2026-07-09', '11:00', 'Pendiente')
ON CONFLICT DO NOTHING;

-- ── RLS: habilitar Row Level Security ────────────────────────
ALTER TABLE patients     ENABLE ROW LEVEL SECURITY;
ALTER TABLE appointments ENABLE ROW LEVEL SECURITY;

-- Política: permitir todo con anon key (ajustar en producción)
CREATE POLICY "allow_all_patients"     ON patients     FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "allow_all_appointments" ON appointments FOR ALL USING (true) WITH CHECK (true);
