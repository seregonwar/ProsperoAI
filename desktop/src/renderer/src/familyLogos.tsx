/*
 * ProsperoAI Desktop — model family logos.
 *
 * Each known model family (Llama, Qwen, Mistral, Gemma, Phi, ...) gets
 * a recognisable tile: brand-tinted gradient + a minimal inline glyph,
 * rendered large at the top of Hub result cards (model-page style).
 * Families are detected from real provider data via shared/families.ts;
 * unknown families fall back to a neutral monogram tile.
 */

import type { ReactNode } from 'react';
import type { HubModel } from '../../shared/types';
import { detectFamilyId, familyLabel } from '../../shared/families';

export interface DetectedFamily {
  known: boolean;
  label: string;
  from: string;
  to: string;
  glyph: ReactNode | null;
}

/* Minimal stroke glyphs (24×24, 1.8 stroke, white) — one per family. */
const GLYPHS: Record<string, ReactNode> = {
  llama: (
    <>
      <path d="M9 5.6 7.1 2.7M15 5.6 16.9 2.7" />
      <path d="M12 6.8c-3.5 0-6 2.6-6 6.2v3.1c0 1.6 1.4 3 3 3h6c1.6 0 3-1.4 3-3V13c0-3.6-2.5-6.2-6-6.2Z" />
      <circle cx="9.9" cy="11.4" r=".9" />
      <circle cx="14.1" cy="11.4" r=".9" />
      <path d="M9.7 14.7c1.4 1.2 3.2 1.2 4.6 0" />
    </>
  ),
  qwen: (
    <>
      <circle cx="10.5" cy="13" r="6" />
      <path d="m15.4 17.6 3.8 3.9" />
    </>
  ),
  mistral: (
    <>
      <path d="M3 8h9.5a2.5 2.5 0 1 0-2.2-3.7" />
      <path d="M3 13h13.5a2.6 2.6 0 1 1-2.3 3.9" />
      <path d="M3 18h6.8a2.3 2.3 0 1 1-2 3.4" />
    </>
  ),
  gemma: (
    <>
      <path d="M7.5 3.5h9L20 8.5l-8 12-8-12 3.5-5Z" />
      <path d="M4 8.5h16" />
      <path d="m12 20.5 3.6-12L12 3.5l-3.6 5 3.6 12Z" />
    </>
  ),
  phi: (
    <>
      <circle cx="11.5" cy="11" r="2.7" />
      <path d="M11.5 8.3V21" />
      <path d="M8.3 4h6.4" />
    </>
  ),
  deepseek: (
    <>
      <circle cx="9.5" cy="13.5" r="5.6" />
      <path d="M15 12.6c2.2 1.5 3.6 3.2 4.4 5.2-.2-2.4.4-4.5 1.6-6.1-1.2-1.6-1.8-3.7-1.6-6.1-.8 2-2.2 3.7-4.4 5.2Z" />
    </>
  ),
  falcon: (
    <>
      <path d="M4 16c4.5-6.5 10-9.8 16-11-1.3 6.2-4.4 10.2-8.8 12.4l-2.3-3.6L5.8 15.6 4 16Z" />
    </>
  ),
  gpt: (
    <>
      <path d="M19 12a7 7 0 1 1-2-4.9" />
      <path d="M12 12h7" />
    </>
  ),
  bert: (
    <>
      <path d="M6.5 4.5h4.8a4 4 0 0 1 0 8H6.5v-8Z" />
      <path d="M6.5 12.5h5.6a4 4 0 0 1 0 8H6.5v-8Z" />
    </>
  ),
  t5: (
    <>
      <path d="M5 5.5h13" />
      <path d="M11.5 5.5V12" />
      <path d="M17.2 9.2h-5.3l-.8 3.6h3.2a2.4 2.4 0 1 1-.5 4.7 2.5 2.5 0 0 1-1.7-1.1" />
    </>
  ),
  yi: (
    <>
      <path d="M4.5 5 12 13.2 19.5 5" />
      <path d="M12 13.2V19" />
    </>
  ),
  grok: (
    <>
      <path d="m13.4 2.5-8.9 11.5h6.5l-1 7.5 8.9-11.5h-6.5l1-7.5Z" />
    </>
  ),
  command: (
    <>
      <path d="M6.5 4.5h6.5a5 5 0 0 1 0 10H6.5v-10Z" />
      <path d="m11 14.5 5 5" />
    </>
  ),
  granite: (
    <>
      <path d="m4 17.5 6-9 4 5.5 3-4 3 7.5H4Z" />
    </>
  ),
  nemotron: (
    <>
      <path d="M6 4.5v15" />
      <path d="M6 4.5 18 19.5v-15" />
    </>
  ),
  olmo: (
    <>
      <circle cx="12" cy="12" r="7" />
    </>
  ),
  starcoder: (
    <>
      <path d="m12 3.5 2.4 5.2 5.6.7-4.1 3.9 1.1 5.6-5-2.8-5 2.8 1.1-5.6-4.1-3.9 5.6-.7L12 3.5Z" />
    </>
  ),
  solar: (
    <>
      <circle cx="12" cy="12" r="3.4" />
      <path d="M12 3.2v1.8M12 19v1.8M3.2 12H5M19 12h1.8M5.6 5.6l1.3 1.3M17.1 17.1l1.3 1.3M18.4 5.6l-1.3 1.3M6.9 17.1l-1.3 1.3" />
    </>
  ),
  mpt: (
    <>
      <path d="M4.5 19.5v-13L12 16l7.5-9.5v13" />
    </>
  ),
  dbrx: (
    <>
      <path d="m12 4 8 15H4l8-15Z" />
      <path d="m12 4 3.2 5.8H8.8L12 4Z" />
    </>
  ),
  glm: (
    <>
      <path d="M19 12a7 7 0 1 1-2-4.9" />
      <path d="M12 12h7" />
    </>
  ),
  internlm: (
    <>
      <path d="M12 4.5v15" />
      <path d="M8 4.5h8M8 19.5h8" />
    </>
  ),
  jamba: (
    <>
      <path d="M8.5 4.5v9a4 4 0 0 0 8 0v-.6" />
      <circle cx="9.6" cy="18.8" r="1" fill="currentColor" stroke="none" />
    </>
  ),
};

/* Brand-tinted gradients, from → to. */
const COLORS: Record<string, [string, string]> = {
  llama: ['#FF9A3D', '#E8440A'],
  qwen: ['#A78BFA', '#5B21B6'],
  mistral: ['#FF7043', '#C62828'],
  gemma: ['#64B5F6', '#1E5FA8'],
  phi: ['#38BDF8', '#075985'],
  deepseek: ['#5C7CFA', '#1E3A8A'],
  falcon: ['#2DD4BF', '#0F766E'],
  gpt: ['#34D399', '#065F46'],
  bert: ['#5C8DF6', '#274B9E'],
  t5: ['#A5B4CB', '#475569'],
  yi: ['#FCA5A5', '#B91C1C'],
  grok: ['#9CA3AF', '#374151'],
  command: ['#5EEAD4', '#115E59'],
  granite: ['#6FA8FF', '#0B3D91'],
  nemotron: ['#8FD14F', '#4E7A00'],
  olmo: ['#818CF8', '#4338CA'],
  starcoder: ['#FBBF24', '#B45309'],
  solar: ['#FDE68A', '#D97706'],
  mpt: ['#FB923C', '#C2410C'],
  dbrx: ['#FF6B5E', '#B31217'],
  glm: ['#60A5FA', '#1D4ED8'],
  internlm: ['#22D3EE', '#0E7490'],
  jamba: ['#A78BFA', '#6D28D9'],
};

const FALLBACK_COLORS: [string, string] = ['#64748B', '#334155'];

/** Readable fallback label from the model id ("SmolLM2-1.7B" -> "SmolLM2"). */
function labelFromId(id: string): string {
  const base = id.split('/').pop() ?? id;
  const head = base.split(/[-_]/)[0];
  return head && /[a-z]/i.test(head) ? head : 'Model';
}

/** Detect the family visual for a Hub model (real provider data). */
export function detectFamily(model: Pick<HubModel, 'id' | 'tags' | 'family'>): DetectedFamily {
  const familyId = detectFamilyId(model.id, model.tags, model.family);
  if (!familyId) {
    return { known: false, label: labelFromId(model.id), from: FALLBACK_COLORS[0], to: FALLBACK_COLORS[1], glyph: null };
  }
  const colors = COLORS[familyId] ?? FALLBACK_COLORS;
  return {
    known: true,
    label: familyLabel(familyId),
    from: colors[0],
    to: colors[1],
    glyph: GLYPHS[familyId] ?? null,
  };
}

export function FamilyLogo({ model, size = 46, radius = 13 }: { model: Pick<HubModel, 'id' | 'tags' | 'family'>; size?: number; radius?: number }) {
  const visual = detectFamily(model);
  return (
    <span
      className="family-logo"
      role="img"
      aria-label={`Model family: ${visual.label}`}
      title={`Family: ${visual.label}`}
      style={{ width: size, height: size, borderRadius: radius, background: `linear-gradient(135deg, ${visual.from}, ${visual.to})` }}
    >
      {visual.glyph
        ? <svg width={Math.round(size * 0.56)} height={Math.round(size * 0.56)} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{visual.glyph}</svg>
        : <span className="family-monogram" style={{ fontSize: Math.round(size * 0.36) }}>{visual.label.slice(0, 1)}</span>}
    </span>
  );
}
