import React, { useState, useEffect, useRef } from 'react';
import { 
  Calendar, Users, BrainCircuit, MessageSquare, 
  Settings, Activity, AlertCircle, CheckCircle2, 
  Send, Bot, User, Search,
  CalendarDays, ChevronRight, UserCircle2
} from 'lucide-react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer
} from 'recharts';

// --- TYPES & MOCK DATABASE ---
type Role = 'admin' | 'paciente';
type Intent = 'agendar' | 'cancelar' | 'consultar' | 'modificar' | 'humano' | 'desconocido';

interface Message {
  id: string;
  sender: 'bot' | 'user' | 'system' | 'admin';
  text: string;
  timestamp: Date;
}

interface Appointment {
  id: string;
  patientName: string;
  dni?: string;
  doctor: string;
  specialty: string;
  date: string;
  time: string;
  status: 'Confirmada' | 'Pendiente' | 'Cancelada' | 'No-Show';
}

type ChatStep = 'idle' | 'asking_specialty' | 'choosing_option' | 'asking_first_name' | 'asking_last_name' | 'asking_dni' | 'ready_to_schedule' | 'handoff' | string;

interface AgentState {
  intent: Intent | null;
  extractedData: any;
  step: ChatStep;
  conversationHistory: { role: string; content: string }[];
}

const mockAppointments: Appointment[] = [
  { id: '1', patientName: 'Carlos Ruiz', dni: '72345678', doctor: 'Dr. Ramírez', specialty: 'Odontología General', date: '2026-07-08', time: '08:00', status: 'Confirmada' },
  { id: '2', patientName: 'Ana Gomez', dni: '45678912', doctor: 'Dra. Luna', specialty: 'Odontopediatría', date: '2026-07-08', time: '10:30', status: 'Pendiente' },
  { id: '3', patientName: 'Luis Merino', dni: '12345678', doctor: 'Dra. Flores', specialty: 'Ortodoncia', date: '2026-07-08', time: '09:00', status: 'No-Show' },
  { id: '4', patientName: 'Sofia Castro', dni: '76543210', doctor: 'Dr. Tello', specialty: 'Endodoncia', date: '2026-07-08', time: '10:00', status: 'Confirmada' },
];

const mockStats = [
  { name: 'Lun', citas: 12, noShows: 2 },
  { name: 'Mar', citas: 19, noShows: 3 },
  { name: 'Mie', citas: 15, noShows: 1 },
  { name: 'Jue', citas: 22, noShows: 4 },
  { name: 'Vie', citas: 18, noShows: 2 },
];

// ── URL del backend LangGraph ─────────────────────────────────────────────
// En desarrollo: http://localhost:8000
// En producción: cambia por tu URL de Railway/Render
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';

// ── Función que llama al backend real (reemplaza simulateAgentGraph) ──────
async function callAgentAPI(
  userInput: string,
  currentState: AgentState,
  addAppointment: (app: Omit<Appointment, 'id' | 'status'>) => void
): Promise<{ response: string; newState: AgentState; refreshData: boolean }> {
  const res = await fetch(`${BACKEND_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message: userInput,
      state: {
        step: currentState.step,
        intent: currentState.intent,
        extracted: currentState.extractedData,
        conversation_history: currentState.conversationHistory ?? [],
      },
    }),
  });

  if (!res.ok) {
    throw new Error(`Error del servidor: ${res.status}`);
  }

  const data = await res.json();

  // Si el backend devolvió una cita nueva, registrarla en el estado React
  if (data.new_appointment) {
    addAppointment(data.new_appointment);
  }

  // Mapear el estado del backend al formato del frontend
  const newState: AgentState = {
    intent: data.state.intent ?? null,
    step: data.state.step ?? 'idle',
    extractedData: data.state.extracted ?? {},
    conversationHistory: data.state.conversation_history ?? [],
  };

  return { 
    response: data.response, 
    newState,
    refreshData: !!data.refresh_data // Capturamos el flag del backend
  };
}

// --- COMPONENTS ---

const AdminDashboard = ({ 
  appointments, 
  deepAgentAlerts, 
  runDeepAgent,
  isDeepAgentRunning,
  liveMessages,
  agentState
}: { 
  appointments: Appointment[], 
  deepAgentAlerts: string[],
  runDeepAgent: () => void,
  isDeepAgentRunning: boolean,
  liveMessages: Message[],
  agentState: AgentState
}) => {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'calendario' | 'pacientes' | 'chats'>('dashboard');

  // Helper to group appointments by date for the calendar view
  const groupedByDate = appointments.reduce((acc, curr) => {
    if (!acc[curr.date]) acc[curr.date] = [];
    acc[curr.date].push(curr);
    return acc;
  }, {} as Record<string, Appointment[]>);

  // Helper to extract unique patients
  const uniquePatients = Array.from(new Set(appointments.map(a => a.dni || a.patientName)))
    .map(id => {
      const apps = appointments.filter(a => (a.dni || a.patientName) === id);
      return {
        id,
        name: apps[0].patientName,
        dni: apps[0].dni || 'No registrado',
        lastSpecialty: apps[0].specialty,
        totalAppointments: apps.length,
        status: apps.some(a => a.status === 'No-Show') ? 'Riesgo Alto' : 'Regular'
      };
    });

  return (
    <div className="flex h-screen bg-slate-50 font-sans text-slate-800">
      {/* Sidebar */}
      <div className="w-64 bg-slate-900 text-white flex flex-col">
        <div className="p-6 text-2xl font-bold tracking-tight text-blue-400 flex items-center gap-2">
          <Activity size={28} />
          Cobba Admin
        </div>
        <nav className="flex-1 px-4 space-y-2 mt-4">
          <button onClick={() => setActiveTab('dashboard')} className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition ${activeTab === 'dashboard' ? 'bg-blue-600/20 text-blue-400' : 'hover:bg-slate-800 text-slate-300'}`}><Activity size={20} /> Dashboard</button>
          <button onClick={() => setActiveTab('calendario')} className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition ${activeTab === 'calendario' ? 'bg-blue-600/20 text-blue-400' : 'hover:bg-slate-800 text-slate-300'}`}><CalendarDays size={20} /> Calendario</button>
          <button onClick={() => setActiveTab('pacientes')} className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition ${activeTab === 'pacientes' ? 'bg-blue-600/20 text-blue-400' : 'hover:bg-slate-800 text-slate-300'}`}><Users size={20} /> Pacientes</button>
          <button onClick={() => setActiveTab('chats')} className={`w-full flex justify-between items-center px-4 py-3 rounded-lg transition ${activeTab === 'chats' ? 'bg-blue-600/20 text-blue-400' : 'hover:bg-slate-800 text-slate-300'}`}>
            <div className="flex items-center gap-3"><MessageSquare size={20} /> Chats Activos</div>
            {agentState.step === 'handoff' && <span className="flex h-3 w-3 relative"><span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-rose-400 opacity-75"></span><span className="relative inline-flex rounded-full h-3 w-3 bg-rose-500"></span></span>}
          </button>
        </nav>
        <div className="p-4 border-t border-slate-800">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-slate-800 flex items-center justify-center"><User size={20} className="text-slate-400"/></div>
            <div className="text-sm">
              <p className="font-medium">Admin Principal</p>
              <p className="text-slate-400 text-xs">Recepción</p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-auto p-8 bg-slate-50/50">
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold text-slate-900 tracking-tight">
            {activeTab === 'dashboard' && 'Panel de Control General'}
            {activeTab === 'calendario' && 'Agenda y Calendario Médicos'}
            {activeTab === 'pacientes' && 'Directorio de Pacientes'}
            {activeTab === 'chats' && 'Monitor de Conversaciones (LangGraph)'}
          </h1>
          <div className="flex gap-3">
            <div className="bg-white border border-slate-200 px-4 py-2 rounded-lg shadow-sm text-sm flex items-center gap-2 text-slate-500">
              <Search size={16} /> Buscar...
            </div>
            <button className="flex items-center gap-2 bg-white border border-slate-200 px-4 py-2 rounded-lg shadow-sm text-sm font-medium hover:bg-slate-50 text-slate-700 transition">
              <Settings size={16} /> Configuración
            </button>
          </div>
        </div>

        {/* ----------------- DASHBOARD TAB ----------------- */}
        {activeTab === 'dashboard' && (
          <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
            {/* KPIs */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
              <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm flex flex-col justify-between">
                <div className="text-slate-500 text-sm font-medium flex justify-between items-center">Asistencia Mensual <CheckCircle2 size={16} className="text-emerald-500"/></div>
                <div className="text-3xl font-bold text-slate-900 mt-2">88%</div>
                <div className="text-xs text-rose-500 mt-2 font-medium bg-rose-50 w-fit px-2 py-0.5 rounded-full">↓ 2% vs mes pasado</div>
              </div>
              <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm flex flex-col justify-between">
                <div className="text-slate-500 text-sm font-medium flex justify-between items-center">Citas Reservadas (Hoy) <Calendar size={16} className="text-blue-500"/></div>
                <div className="text-3xl font-bold text-slate-900 mt-2">{appointments.length}</div>
                <div className="text-xs text-emerald-600 mt-2 font-medium bg-emerald-50 w-fit px-2 py-0.5 rounded-full">Actualizado en vivo</div>
              </div>
              <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm flex flex-col justify-between">
                <div className="text-slate-500 text-sm font-medium flex justify-between items-center">Pacientes Atendidos <Users size={16} className="text-indigo-500"/></div>
                <div className="text-3xl font-bold text-slate-900 mt-2">1,204</div>
                <div className="text-xs text-slate-500 mt-2 font-medium">Este año</div>
              </div>
              <div className="bg-gradient-to-br from-indigo-700 to-blue-800 p-6 rounded-xl shadow-md text-white flex flex-col justify-between relative overflow-hidden">
                <div className="absolute top-0 right-0 p-4 opacity-20"><BrainCircuit size={64}/></div>
                <div className="text-indigo-100 text-sm font-medium relative z-10">Agente Analítico (DeepAgent)</div>
                <div className="text-2xl font-bold mb-3 relative z-10">En espera</div>
                <button onClick={runDeepAgent} disabled={isDeepAgentRunning} className="relative z-10 text-xs bg-white/20 hover:bg-white/30 backdrop-blur-sm transition px-4 py-2 rounded-lg font-medium w-fit flex items-center gap-2 disabled:opacity-60 disabled:cursor-not-allowed">
                  <Activity size={14}/> {isDeepAgentRunning ? 'Analizando...' : 'Ejecutar Análisis'}
                </button>
              </div>
            </div>

            {/* Deep Agent Alerts */}
            {deepAgentAlerts.length > 0 && (
              <div className="mb-8 animate-in fade-in duration-300">
                {deepAgentAlerts.map((alert, index) => (
                  <div key={index} className="bg-indigo-50 p-5 rounded-xl border border-indigo-200 mb-3 flex gap-4 items-start shadow-sm">
                    <div className="bg-indigo-100 p-2 rounded-full shrink-0">
                       <BrainCircuit size={20} className="text-indigo-600" />
                    </div>
                    <div>
                      <h4 className="font-bold text-indigo-900 text-sm mb-1">Insight Generado por IA</h4>
                      <p className="text-indigo-800 text-sm leading-relaxed">{alert}</p>
                      <div className="mt-3 flex gap-2">
                        <button className="text-xs bg-indigo-600 text-white px-4 py-1.5 rounded-md hover:bg-indigo-700 transition font-medium shadow-sm">Aplicar Automatización</button>
                        <button className="text-xs bg-white text-indigo-700 border border-indigo-200 px-4 py-1.5 rounded-md hover:bg-indigo-50 transition font-medium">Descartar</button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              {/* Quick View Citas */}
              <div className="lg:col-span-2 bg-white rounded-xl border border-slate-200 shadow-sm flex flex-col h-[400px]">
                <div className="p-6 border-b border-slate-100 flex justify-between items-center">
                  <h2 className="font-bold text-slate-800">Últimas Citas Registradas</h2>
                  <button onClick={() => setActiveTab('calendario')} className="text-blue-600 text-sm font-medium hover:underline flex items-center">Ver calendario <ChevronRight size={16}/></button>
                </div>
                <div className="overflow-auto flex-1 p-2">
                  <table className="w-full text-left text-sm">
                    <thead className="text-slate-400 font-medium sticky top-0 bg-white/90 backdrop-blur-sm z-10">
                      <tr>
                        <th className="p-4">Paciente</th>
                        <th className="p-4">Médico</th>
                        <th className="p-4">Fecha/Hora</th>
                        <th className="p-4">Estado</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-50">
                      {appointments.slice(-5).reverse().map(app => (
                        <tr key={app.id} className="hover:bg-slate-50 transition group">
                          <td className="p-4">
                            <div className="font-medium text-slate-800">{app.patientName}</div>
                            <div className="text-xs text-slate-400 mt-0.5">{app.specialty}</div>
                          </td>
                          <td className="p-4 text-slate-600">{app.doctor}</td>
                          <td className="p-4 text-slate-600 font-medium">{app.date} <span className="text-slate-400 font-normal">a las {app.time}</span></td>
                          <td className="p-4">
                            <span className={`px-2.5 py-1 rounded-md text-xs font-medium border ${
                              app.status === 'Confirmada' ? 'bg-emerald-50 text-emerald-700 border-emerald-200' :
                              app.status === 'Pendiente' ? 'bg-amber-50 text-amber-700 border-amber-200' :
                              app.status === 'No-Show' ? 'bg-rose-50 text-rose-700 border-rose-200' :
                              'bg-slate-50 text-slate-700 border-slate-200'
                            }`}>
                              {app.status}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Chart */}
              <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm h-[400px] flex flex-col">
                 <h2 className="font-bold text-slate-800 mb-6">Tráfico Semanal</h2>
                 <div className="flex-1">
                   <ResponsiveContainer width="100%" height="100%">
                     <BarChart data={mockStats} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
                       <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                       <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{fill: '#94a3b8', fontSize: 12}} />
                       <YAxis axisLine={false} tickLine={false} tick={{fill: '#94a3b8', fontSize: 12}} />
                       <RechartsTooltip cursor={{fill: '#f8fafc'}} contentStyle={{borderRadius: '8px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)'}}/>
                       <Legend wrapperStyle={{fontSize: '12px', paddingTop: '10px'}}/>
                       <Bar dataKey="citas" name="Citas" fill="#3b82f6" radius={[4, 4, 0, 0]} barSize={24} />
                       <Bar dataKey="noShows" name="Ausencias" fill="#fb7185" radius={[4, 4, 0, 0]} barSize={24} />
                     </BarChart>
                   </ResponsiveContainer>
                 </div>
              </div>
            </div>
          </div>
        )}

        {/* ----------------- CALENDARIO TAB ----------------- */}
        {activeTab === 'calendario' && (
          <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 flex flex-col gap-6">
             {Object.entries(groupedByDate).sort((a,b) => a[0].localeCompare(b[0])).map(([date, dayAppointments]) => (
               <div key={date} className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
                 <div className="bg-slate-100/50 px-6 py-4 border-b border-slate-200 flex items-center gap-3">
                    <CalendarDays size={20} className="text-blue-600"/>
                    <h3 className="font-bold text-slate-800 capitalize text-lg">{date}</h3>
                    <span className="ml-2 bg-blue-100 text-blue-800 text-xs px-2 py-0.5 rounded-full font-medium">{dayAppointments.length} citas</span>
                 </div>
                 <div className="divide-y divide-slate-100">
                    {dayAppointments.sort((a,b) => a.time.localeCompare(b.time)).map(app => (
                      <div key={app.id} className="p-6 flex flex-col sm:flex-row sm:items-center justify-between gap-4 hover:bg-slate-50 transition">
                         <div className="flex items-start gap-4">
                            <div className="bg-blue-50 text-blue-700 rounded-lg p-3 text-center min-w-[90px] border border-blue-100">
                               <div className="text-sm font-bold">{app.time}</div>
                            </div>
                            <div>
                               <h4 className="text-lg font-bold text-slate-800 flex items-center gap-2">
                                 {app.patientName} 
                                 <span className={`px-2 py-0.5 rounded text-[10px] uppercase tracking-wider font-bold border ${app.status === 'Confirmada' ? 'bg-emerald-50 text-emerald-600 border-emerald-200' : 'bg-slate-100 text-slate-500 border-slate-200'}`}>{app.status}</span>
                               </h4>
                               <p className="text-slate-500 text-sm mt-1 flex items-center gap-2">
                                  <UserCircle2 size={16} /> DNI: {app.dni || 'Pendiente'}
                               </p>
                            </div>
                         </div>
                         <div className="bg-slate-50 border border-slate-200 rounded-lg px-4 py-3 min-w-[200px]">
                           <p className="text-xs text-slate-500 font-medium uppercase tracking-wide">Asignado a</p>
                           <p className="font-bold text-slate-800 mt-1">{app.doctor}</p>
                           <p className="text-sm text-blue-600 mt-0.5">{app.specialty}</p>
                         </div>
                      </div>
                    ))}
                 </div>
               </div>
             ))}
          </div>
        )}

        {/* ----------------- PACIENTES TAB ----------------- */}
        {activeTab === 'pacientes' && (
          <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
               <div className="p-6 border-b border-slate-200 flex justify-between items-center bg-slate-50">
                  <h2 className="font-bold text-slate-800">Directorio General ({uniquePatients.length} registrados)</h2>
                  <button className="bg-blue-600 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-blue-700 transition shadow-sm">
                    Exportar Excel
                  </button>
               </div>
               <table className="w-full text-left text-sm">
                  <thead className="bg-white text-slate-500 font-medium border-b border-slate-200">
                    <tr>
                      <th className="p-5">Nombre y DNI</th>
                      <th className="p-5">Última Especialidad</th>
                      <th className="p-5">Historial</th>
                      <th className="p-5">Alerta / Estado</th>
                      <th className="p-5">Acciones</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                     {uniquePatients.map((patient, idx) => (
                       <tr key={idx} className="hover:bg-slate-50 transition">
                         <td className="p-5">
                           <div className="font-bold text-slate-800 text-base">{patient.name}</div>
                           <div className="text-slate-500 mt-1 flex items-center gap-1.5"><span className="w-3 h-3 rounded border border-slate-300 inline-block bg-slate-100"></span> DNI: {patient.dni}</div>
                         </td>
                         <td className="p-5 text-slate-700 font-medium">{patient.lastSpecialty}</td>
                         <td className="p-5">
                           <span className="bg-slate-100 text-slate-700 px-3 py-1 rounded-full text-xs font-bold border border-slate-200">
                             {patient.totalAppointments} citas registradas
                           </span>
                         </td>
                         <td className="p-5">
                            {patient.status === 'Riesgo Alto' ? (
                               <span className="flex items-center gap-1.5 text-rose-600 text-xs font-bold bg-rose-50 px-2.5 py-1.5 rounded-md border border-rose-200 w-fit">
                                 <AlertCircle size={14} /> Antecedente de No-Show
                               </span>
                            ) : (
                               <span className="text-emerald-600 text-xs font-bold bg-emerald-50 px-2.5 py-1.5 rounded-md border border-emerald-200 w-fit flex items-center gap-1.5">
                                 <CheckCircle2 size={14} /> Paciente Regular
                               </span>
                            )}
                         </td>
                         <td className="p-5">
                            <button className="text-blue-600 font-medium hover:underline">Ver ficha</button>
                         </td>
                       </tr>
                     ))}
                  </tbody>
               </table>
            </div>
          </div>
        )}

        {/* ----------------- CHATS ACTIVOS TAB (HANDOFF) ----------------- */}
        {activeTab === 'chats' && (
          <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 h-[700px] flex gap-6">
            
            {/* Inbox List */}
            <div className="w-1/3 bg-white rounded-xl border border-slate-200 shadow-sm flex flex-col overflow-hidden">
               <div className="p-4 border-b border-slate-200 bg-slate-50">
                 <h2 className="font-bold text-slate-800">Bandeja de Entrada</h2>
               </div>
               <div className="overflow-y-auto flex-1">
                 {/* Live Chat Item */}
                 <div className={`p-4 border-b border-slate-100 cursor-pointer transition ${agentState.step === 'handoff' ? 'bg-rose-50 border-l-4 border-l-rose-500' : 'hover:bg-slate-50 border-l-4 border-l-transparent'}`}>
                    <div className="flex justify-between items-start mb-1">
                      <h4 className="font-bold text-slate-800 text-sm">Visitante Anónimo</h4>
                      <span className="text-xs text-slate-400">Ahora</span>
                    </div>
                    <p className="text-xs text-slate-500 truncate mb-2">Último msg: "{liveMessages[liveMessages.length-1]?.text.substring(0, 30)}..."</p>
                    {agentState.step === 'handoff' ? (
                       <span className="inline-flex items-center gap-1 bg-rose-100 text-rose-700 text-[10px] font-bold px-2 py-0.5 rounded border border-rose-200 uppercase tracking-wider">
                         <AlertCircle size={12}/> Requiere Atención
                       </span>
                    ) : (
                       <span className="inline-flex items-center gap-1 bg-emerald-100 text-emerald-700 text-[10px] font-bold px-2 py-0.5 rounded border border-emerald-200 uppercase tracking-wider">
                         <Bot size={12}/> Atendido por IA
                       </span>
                    )}
                 </div>
                 
                 {/* Mock past chats */}
                 <div className="p-4 border-b border-slate-100 opacity-60">
                    <div className="flex justify-between items-start mb-1">
                      <h4 className="font-bold text-slate-800 text-sm">María López</h4>
                      <span className="text-xs text-slate-400">Hace 2h</span>
                    </div>
                    <p className="text-xs text-slate-500 truncate mb-2">Consulta resuelta (Ortodoncia)</p>
                    <span className="inline-flex items-center gap-1 bg-slate-100 text-slate-600 text-[10px] font-bold px-2 py-0.5 rounded border border-slate-200 uppercase tracking-wider">
                      Cerrado
                    </span>
                 </div>
               </div>
            </div>

            {/* Chat View */}
            <div className="flex-1 bg-white rounded-xl border border-slate-200 shadow-sm flex flex-col overflow-hidden relative">
               <div className="p-4 border-b border-slate-200 bg-white flex justify-between items-center z-10 shadow-sm">
                 <div>
                    <h3 className="font-bold text-slate-800">Chat en Vivo #4829</h3>
                    <p className="text-xs text-slate-500">Conectado vía Widget Web</p>
                 </div>
                 {agentState.step === 'handoff' && (
                    <button className="bg-rose-600 text-white px-4 py-2 rounded-lg text-sm font-bold shadow-sm hover:bg-rose-700 transition flex items-center gap-2">
                       <UserCircle2 size={16}/> Tomar control (Handoff)
                    </button>
                 )}
               </div>
               
               <div className="flex-1 bg-slate-50 overflow-y-auto p-6 space-y-4">
                  <div className="text-center text-xs text-slate-400 mb-6 bg-white py-1 px-4 rounded-full border border-slate-200 w-fit mx-auto shadow-sm">
                    Historial sincronizado desde LangGraph (Redis)
                  </div>
                  {liveMessages.map((msg, i) => (
                    <div key={i} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                      {msg.sender === 'bot' && (
                         <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center shrink-0 mr-3 mt-1 shadow-sm"><Bot size={16} className="text-white"/></div>
                      )}
                      <div className={`max-w-[70%] rounded-2xl px-5 py-3 text-sm shadow-sm ${
                        msg.sender === 'user' 
                          ? 'bg-slate-800 text-white rounded-tr-none' 
                          : 'bg-white border border-slate-200 text-slate-700 rounded-tl-none leading-relaxed whitespace-pre-wrap'
                      }`}>
                        {msg.text}
                      </div>
                    </div>
                  ))}
               </div>

               <div className="p-4 bg-white border-t border-slate-200">
                  <div className="flex gap-2 opacity-50 cursor-not-allowed">
                     <input disabled type="text" placeholder="Escribe un mensaje al paciente..." className="flex-1 bg-slate-100 border border-slate-200 rounded-lg px-4 py-2.5 text-sm" />
                     <button disabled className="bg-slate-300 text-white px-4 rounded-lg"><Send size={18}/></button>
                  </div>
                  {agentState.step !== 'handoff' && (
                    <p className="text-xs text-slate-400 text-center mt-2">El Agente IA está gestionando esta conversación actualmente.</p>
                  )}
               </div>
            </div>
          </div>
        )}

      </div>
    </div>
  );
};

const PatientChat = ({ 
  messages, 
  onSendMessage, 
  isTyping 
}: { 
  messages: Message[], 
  onSendMessage: (txt: string) => void,
  isTyping: boolean 
}) => {
  const [input, setInput] = useState('');
  const endOfMessagesRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endOfMessagesRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  const handleSend = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;
    onSendMessage(input);
    setInput('');
  };

  return (
    <div className="min-h-screen bg-slate-100 flex items-center justify-center font-sans p-4 relative overflow-hidden">
      {/* Background Decor */}
      <div className="absolute top-0 left-0 w-full h-96 bg-blue-600 rounded-b-[40%] shadow-lg opacity-90 pointer-events-none"></div>
      
      <div className="z-10 text-center text-white mb-10 absolute top-8">
        <h1 className="text-4xl font-extrabold tracking-tight mb-2 flex items-center justify-center gap-3 drop-shadow-md">
          <Activity size={36} /> Clínica Cobba
        </h1>
        <p className="text-blue-100 text-lg font-medium">Tu salud, más accesible que nunca.</p>
      </div>

      {/* Chat Widget Container */}
      <div className="bg-white w-full max-w-md rounded-2xl shadow-2xl overflow-hidden flex flex-col h-[650px] border border-slate-200 mt-20 z-10 relative">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-600 to-blue-700 p-4 text-white flex justify-between items-center shadow-md z-20">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 bg-white/20 backdrop-blur-sm rounded-full flex items-center justify-center shadow-inner border border-white/30">
              <Bot size={28} className="text-white" />
            </div>
            <div>
              <h3 className="font-bold text-lg leading-tight tracking-wide">Asistente Virtual</h3>
              <p className="text-blue-100 text-xs flex items-center gap-1.5 font-medium mt-0.5">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-300 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-400"></span>
                </span>
                En línea
              </p>
            </div>
          </div>
        </div>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto p-5 space-y-5 bg-slate-50/50">
          <div className="text-center text-xs font-medium text-slate-400 my-2 bg-white w-fit mx-auto px-4 py-1 rounded-full border border-slate-100 shadow-sm">Hoy, {new Date().toLocaleDateString()}</div>
          
          {messages.map((msg) => (
            <div key={msg.id} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[85%] rounded-2xl px-5 py-3 text-[15px] shadow-sm leading-relaxed whitespace-pre-wrap ${
                msg.sender === 'user' 
                  ? 'bg-blue-600 text-white rounded-tr-none' 
                  : msg.sender === 'system'
                  ? 'bg-slate-800 text-white w-full text-center text-xs opacity-70'
                  : 'bg-white border border-slate-200 text-slate-700 rounded-tl-none'
              }`}>
                {msg.text}
              </div>
            </div>
          ))}

          {isTyping && (
             <div className="flex justify-start">
               <div className="bg-white border border-slate-200 rounded-2xl rounded-tl-none px-5 py-4 shadow-sm flex gap-1.5 items-center w-fit">
                 <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"></div>
                 <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s'}}></div>
                 <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0.4s'}}></div>
               </div>
             </div>
          )}
          <div ref={endOfMessagesRef} />
        </div>

        {/* Input Area */}
        <form onSubmit={handleSend} className="p-3 bg-white border-t border-slate-100 shadow-[0_-4px_6px_-1px_rgba(0,0,0,0.05)]">
          <div className="mt-1 mb-3 flex gap-2 overflow-x-auto pb-2 px-1 no-scrollbar">
             {/* Quick Replies */}
             <button type="button" onClick={() => onSendMessage("Quiero agendar una cita")} className="whitespace-nowrap text-xs font-bold bg-blue-50 text-blue-700 px-4 py-2 rounded-full border border-blue-200 hover:bg-blue-100 transition shadow-sm">Agendar cita</button>
             <button type="button" onClick={() => onSendMessage("Necesito hablar con un humano")} className="whitespace-nowrap text-xs font-bold bg-slate-50 text-slate-700 px-4 py-2 rounded-full border border-slate-200 hover:bg-slate-100 transition shadow-sm">Hablar con humano</button>
          </div>
          <div className="flex gap-2">
            <input 
              type="text" 
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Escribe tu mensaje aquí..."
              className="flex-1 bg-slate-100 border border-transparent rounded-full px-5 py-3 text-sm focus:outline-none focus:bg-white focus:border-blue-500 focus:ring-2 focus:ring-blue-100 transition"
            />
            <button 
              type="submit" 
              disabled={!input.trim()}
              className="bg-blue-600 text-white p-3 rounded-full hover:bg-blue-700 disabled:opacity-50 transition shadow-md flex items-center justify-center shrink-0"
            >
              <Send size={18} className="ml-0.5" />
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default function App() {
  const [view, setView] = useState<Role>('paciente');
  
  // App State — citas cargadas desde Supabase
  const [appointments, setAppointments] = useState<Appointment[]>([]);
  const [, setLoadingAppointments] = useState(true);
  const [deepAgentAlerts, setDeepAgentAlerts] = useState<string[]>([]);

  // 1. Extraemos la lógica a una función para poder re-usarla
  const fetchAppointments = async () => {
    setLoadingAppointments(true);
    try {
      const res = await fetch(`${BACKEND_URL}/appointments`);
      if (!res.ok) throw new Error('Error al cargar citas');
      const data = await res.json();
      
      const mapped: Appointment[] = data.map((a: any) => ({
        id: a.id,
        patientName: a.patient_name,
        dni: a.dni,
        doctor: a.doctor,
        specialty: a.specialty,
        date: a.date,
        time: a.time,
        status: a.status,
      }));
      setAppointments(mapped);
    } catch {
      setAppointments(mockAppointments);
    } finally {
      setLoadingAppointments(false);
    }
  };

  // Cargar citas desde Supabase al montar
  useEffect(() => {
    fetchAppointments();
  }, []);
  
  // Chat State
  const [messages, setMessages] = useState<Message[]>([
    { id: '1', sender: 'bot', text: '¡Hola! Soy el asistente virtual de la Clínica Cobba. ¿En qué te puedo ayudar hoy?', timestamp: new Date() }
  ]);
  const [isTyping, setIsTyping] = useState(false);
  const [agentState, setAgentState] = useState<AgentState>({ intent: null, extractedData: {}, step: 'idle', conversationHistory: [] });

  // Deep Agent — llama al backend real para análisis de patrones
  const [isDeepAgentRunning, setIsDeepAgentRunning] = useState(false);

  const handleRunDeepAgent = async () => {
    setIsDeepAgentRunning(true);
    try {
      const res = await fetch(`${BACKEND_URL}/deep-agent`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          appointments: appointments,
          stats: mockStats,
        }),
      });
      if (!res.ok) throw new Error(`Error ${res.status}`);
      const data = await res.json();
      setDeepAgentAlerts(prev => [...prev, ...data.alerts]);
    } catch {
      setDeepAgentAlerts(prev => [
        ...prev,
        '⚠️ No se pudo conectar con el Deep Agent. Verifica que el backend esté corriendo.'
      ]);
    } finally {
      setIsDeepAgentRunning(false);
    }
  };

  // Handle incoming messages from patient — llama al backend LangGraph real
  const handleUserMessage = async (text: string) => {
    const newUserMsg: Message = { id: Date.now().toString(), sender: 'user', text, timestamp: new Date() };
    setMessages(prev => [...prev, newUserMsg]);
    setIsTyping(true);

    try {
      // 2. Aquí recibimos el nuevo flag refreshData
      const { response, newState, refreshData } = await callAgentAPI(
        text,
        agentState,
        async (newApp) => {
          try {
            // Guardar en Supabase
            const res = await fetch(`${BACKEND_URL}/appointments`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                patient_name: newApp.patientName,
                dni: newApp.dni,
                doctor: newApp.doctor,
                specialty: newApp.specialty,
                date: newApp.date,
                time: newApp.time,
                status: 'Confirmada',
              }),
            });
            if (res.ok) {
              const saved = await res.json();
              const appointment: Appointment = {
                id: saved.id,
                patientName: saved.patient_name,
                dni: saved.dni,
                doctor: saved.doctor,
                specialty: saved.specialty,
                date: saved.date,
                time: saved.time,
                status: saved.status,
              };
              setAppointments(prev => [...prev, appointment]);
            } else {
              const errBody = await res.json().catch(() => ({}));
              console.error('No se pudo guardar la cita en Supabase:', res.status, errBody);
              setMessages(prev => [
                ...prev,
                {
                  id: (Date.now() + 1).toString(),
                  sender: 'bot',
                  text: '⚠️ Tu cita no se pudo guardar en el sistema (error del servidor). Por favor intenta de nuevo o contacta a recepción.',
                  timestamp: new Date(),
                },
              ]);
            }
          } catch (err) {
            console.error('Error de red al guardar la cita:', err);
            setMessages(prev => [
              ...prev,
              {
                id: (Date.now() + 1).toString(),
                sender: 'bot',
                text: '⚠️ No se pudo conectar con el servidor para guardar tu cita. Verifica tu conexión e intenta de nuevo.',
                timestamp: new Date(),
              },
            ]);
          }
        }
      );
      
      setAgentState(newState);
      setMessages(prev => [
        ...prev,
        { id: Date.now().toString(), sender: 'bot', text: response, timestamp: new Date() },
      ]);

      // 3. Si el agente completó una modificación/cancelación, refrescamos la BD en la interfaz
      if (refreshData) {
        await fetchAppointments();
      }

    } catch (error) {
      setMessages(prev => [
        ...prev,
        {
          id: Date.now().toString(),
          sender: 'bot',
          text: '⚠️ No pude conectarme con el servidor. Asegúrate de que el backend esté corriendo en http://localhost:8000',
          timestamp: new Date(),
        },
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <div className="relative min-h-screen bg-slate-900">
      {/* View Switcher Overlay (For demo purposes only) */}
      <div className="absolute top-4 right-4 z-50 bg-white/10 backdrop-blur-md border border-white/20 p-2 rounded-xl flex gap-2 shadow-2xl">
        <button 
          onClick={() => setView('paciente')}
          className={`px-4 py-2 rounded-lg text-sm font-bold transition ${view === 'paciente' ? 'bg-blue-600 text-white shadow-md' : 'text-slate-300 hover:bg-white/10'}`}
        >
          Vista Paciente
        </button>
        <button 
          onClick={() => setView('admin')}
          className={`px-4 py-2 rounded-lg text-sm font-bold transition flex gap-2 items-center ${view === 'admin' ? 'bg-slate-800 text-white shadow-md' : 'text-slate-300 hover:bg-white/10'}`}
        >
          Vista Admin 
          {appointments.length > mockAppointments.length && <span className="w-2 h-2 rounded-full bg-rose-500 animate-pulse"></span>}
        </button>
      </div>

      {/* Render Current View */}
      {view === 'paciente' ? (
        <PatientChat 
          messages={messages} 
          onSendMessage={handleUserMessage} 
          isTyping={isTyping} 
        />
      ) : (
        <AdminDashboard 
          appointments={appointments} 
          deepAgentAlerts={deepAgentAlerts}
          runDeepAgent={handleRunDeepAgent}
          isDeepAgentRunning={isDeepAgentRunning}
          liveMessages={messages}
          agentState={agentState}
        />
      )}
    </div>
  );
}