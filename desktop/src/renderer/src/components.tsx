import type { ReactNode } from 'react';

export type IconName =
  | 'grid' | 'layers' | 'console' | 'spark' | 'chart' | 'pulse' | 'settings'
  | 'plus' | 'arrow' | 'search' | 'more' | 'check' | 'bolt' | 'download'
  | 'send' | 'folder' | 'refresh' | 'shield' | 'terminal' | 'copy' | 'chevron'
  | 'trash' | 'stop' | 'wifi' | 'info';

export function Icon({ name, size = 18, className }: { name: IconName; size?: number; className?: string }) {
  const common = {
    width: size, height: size, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor',
    strokeWidth: 1.8, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const, 'aria-hidden': true,
  };
  const paths: Record<IconName, ReactNode> = {
    grid: <><rect x="3" y="3" width="7" height="7" rx="1" /><rect x="14" y="3" width="7" height="7" rx="1" /><rect x="3" y="14" width="7" height="7" rx="1" /><rect x="14" y="14" width="7" height="7" rx="1" /></>,
    layers: <><path d="m12 3 9 5-9 5-9-5 9-5Z" /><path d="m3 12 9 5 9-5" /><path d="m3 16 9 5 9-5" /></>,
    console: <><rect x="3" y="4" width="18" height="14" rx="2" /><path d="M8 21h8M12 18v3M7 9h.01M11 9h.01M15 10h3M16.5 8.5v3" /></>,
    spark: <><path d="m12 3-1.3 5.7L5 10l5.7 1.3L12 17l1.3-5.7L19 10l-5.7-1.3L12 3Z" /><path d="m19 16-.6 2.4L16 19l2.4.6L19 22l.6-2.4L22 19l-2.4-.6L19 16Z" /></>,
    chart: <><path d="M4 19V5M4 19h17" /><path d="m7 15 3-4 3 2 5-7" /><circle cx="7" cy="15" r="1" /><circle cx="10" cy="11" r="1" /><circle cx="13" cy="13" r="1" /><circle cx="18" cy="6" r="1" /></>,
    pulse: <><path d="M3 12h4l2-7 4 14 2-7h6" /></>,
    settings: <><circle cx="12" cy="12" r="3.5" /><path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9l2.1 2.1M17 17l2.1 2.1M19.1 4.9 17 7M7 17l-2.1 2.1" /></>,
    plus: <><path d="M12 5v14M5 12h14" /></>,
    arrow: <><path d="M5 12h14M13 6l6 6-6 6" /></>,
    search: <><circle cx="10.8" cy="10.8" r="6.8" /><path d="m16 16 5 5" /></>,
    more: <><circle cx="5" cy="12" r="1" fill="currentColor" stroke="none" /><circle cx="12" cy="12" r="1" fill="currentColor" stroke="none" /><circle cx="19" cy="12" r="1" fill="currentColor" stroke="none" /></>,
    check: <><path d="m5 12 4 4L19 6" /></>,
    bolt: <><path d="m13 2-9 12h7l-1 8 9-12h-7l1-8Z" /></>,
    download: <><path d="M12 3v12M7 10l5 5 5-5M5 21h14" /></>,
    send: <><path d="m22 2-7 20-4-9-9-4 20-7Z" /><path d="M22 2 11 13" /></>,
    folder: <><path d="M3 7a2 2 0 0 1 2-2h5l2 2h7a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7Z" /></>,
    refresh: <><path d="M20 11a8 8 0 0 0-14.7-3L3 11" /><path d="M3 5v6h6" /><path d="M4 13a8 8 0 0 0 14.7 3L21 13" /><path d="M21 19v-6h-6" /></>,
    shield: <><path d="M12 3 20 6v5c0 5-3.4 8.7-8 10-4.6-1.3-8-5-8-10V6l8-3Z" /><path d="m9 12 2 2 4-4" /></>,
    terminal: <><path d="m5 7 5 5-5 5M13 17h6" /></>,
    copy: <><rect x="9" y="9" width="11" height="11" rx="2" /><path d="M15 9V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v7a2 2 0 0 0 2 2h3" /></>,
    chevron: <><path d="m9 18 6-6-6-6" /></>,
    trash: <><path d="M4 7h16M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2M6 7l1 13h10l1-13M10 11v6M14 11v6" /></>,
    stop: <><rect x="6" y="6" width="12" height="12" rx="2" /></>,
    wifi: <><path d="M2.5 8.5a15 15 0 0 1 19 0M5.5 12a10 10 0 0 1 13 0M8.8 15.5a5 5 0 0 1 6.4 0" /><circle cx="12" cy="19" r="1" fill="currentColor" stroke="none" /></>,
    info: <><circle cx="12" cy="12" r="9" /><path d="M12 11v5M12 8h.01" /></>,
  };
  return <svg {...common} className={className}>{paths[name]}</svg>;
}

export function StatusDot({ status }: { status: string }) {
  const lower = status.toLowerCase();
  return <span className={`status-dot ${lower}`} aria-label={status} />;
}

export function Sparkline({ values, color = '#9a7cff', fill = true }: { values: number[]; color?: string; fill?: boolean }) {
  if (values.length < 2) return <svg className="sparkline" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true" />;
  const max = Math.max(...values);
  const min = Math.min(...values);
  const points = values.map((value, index) => {
    const x = (index / (values.length - 1)) * 100;
    const y = 86 - ((value - min) / Math.max(max - min, 1)) * 66;
    return `${x},${y}`;
  }).join(' ');
  return <svg className="sparkline" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
    {fill && <polygon points={`0,100 ${points} 100,100`} fill={color} opacity=".1" />}
    <polyline points={points} fill="none" stroke={color} strokeWidth="2.4" vectorEffect="non-scaling-stroke" />
  </svg>;
}

export function SectionHeader({ eyebrow, title, description, action }: {
  eyebrow: string; title: string; description?: string; action?: ReactNode;
}) {
  return <div className="section-header"><div><div className="eyebrow">{eyebrow}</div><h1>{title}</h1>{description && <p>{description}</p>}</div>{action}</div>;
}

export function MetricCard({ label, value, detail, color, icon }: {
  label: string; value: string; detail: string; color: string; icon: IconName;
}) {
  return <article className="metric-card">
    <div className="metric-heading"><span className="icon-box" style={{ color, background: `${color}17` }}><Icon name={icon} size={17} /></span><span>{label}</span></div>
    <div className="metric-value">{value}</div>
    <div className="metric-footer"><span>{detail}</span></div>
  </article>;
}

export function EmptyState({ icon, title, description, action }: {
  icon: IconName; title: string; description: string; action?: ReactNode;
}) {
  return <div className="empty-state">
    <div className="empty-state-icon"><Icon name={icon} size={22} /></div>
    <strong>{title}</strong>
    <p>{description}</p>
    {action}
  </div>;
}

export function Toggle({ label, description, enabled = false, onChange }: {
  label: string; description: string; enabled?: boolean; onChange?: (next: boolean) => void;
}) {
  return <div className="toggle-row"><div><strong>{label}</strong><span>{description}</span></div>
    <button className={`toggle ${enabled ? 'on' : ''}`} onClick={() => onChange?.(!enabled)} aria-label={label}><i /></button></div>;
}

/* ---- formatting helpers ---- */

export function formatBytes(bytes?: number): string {
  if (!bytes || bytes <= 0) return '—';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const index = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
  return `${(bytes / 1024 ** index).toFixed(index > 1 ? 1 : 0)} ${units[index]}`;
}

export function formatCount(value?: number): string {
  if (!value) return '—';
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`;
  return String(value);
}

export function formatMs(value?: number): string {
  if (value === undefined) return '—';
  return value < 10 ? `${value.toFixed(2)} ms` : `${Math.round(value)} ms`;
}
