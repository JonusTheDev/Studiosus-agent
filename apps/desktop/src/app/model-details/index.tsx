import { useStore } from '@nanostores/react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useMemo, useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { GlyphSpinner } from '@/components/ui/glyph-spinner'
import { getModelDyno, setGlobalModel } from '@/hermes'
import { useI18n } from '@/i18n'
import { ChevronDown, ChevronRight } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $currentModel, $currentProvider } from '@/store/session'
import type { DynoCurvePoint, DynoModelRow, DynoReport } from '@/types/hermes'

const REFRESH_MS = 5000

function fmtTokS(v: number | null | undefined): string {
  return v == null ? '—' : `${v.toFixed(0)} tok/s`
}

function fmtCtx(v: number | null | undefined): string {
  if (v == null) {return '—'}

  return v % 1024 === 0 ? `${v / 1024}K` : `${v.toLocaleString()}`
}

function pct(v: number | null | undefined): string {
  return v == null ? '—' : `${Math.round(v * 100)}%`
}

/** A tiny inline sparkline of recent tok/s, drawn as an SVG polyline. */
function Sparkline({ points }: { points: number[] }) {
  if (points.length < 2) {return null}
  const w = 48
  const h = 14
  const min = Math.min(...points)
  const max = Math.max(...points)
  const span = max - min || 1
  const step = w / (points.length - 1)

  const d = points
    .map((p, i) => `${(i * step).toFixed(1)},${(h - ((p - min) / span) * h).toFixed(1)}`)
    .join(' ')

  return (
    <svg aria-hidden className="text-(--ui-text-tertiary)" height={h} viewBox={`0 0 ${w} ${h}`} width={w}>
      <polyline fill="none" points={d} stroke="currentColor" strokeWidth={1} />
    </svg>
  )
}

/** The running-process header — a Flame inspection of the active model. */
function RunningProcess({ report }: { report: DynoReport }) {
  const { t } = useI18n()
  const copy = t.modelDetails
  const flame = report.flame

  if (!report.reachable) {
    return <div className="px-3 py-2 text-xs text-muted-foreground">{copy.unreachable}</div>
  }

  if (!flame) {return null}

  const chips: string[] = []

  if (flame.loaded) {chips.push(copy.loaded)}
  else {chips.push(copy.notLoaded)}

  if (flame.pinned) {chips.push(copy.pinned)}

  if (flame.gpu_frac != null) {chips.push(`${pct(flame.gpu_frac)} ${copy.onGpu}`)}

  if (flame.loaded_num_ctx) {chips.push(`${copy.ctx} ${fmtCtx(flame.loaded_num_ctx)}`)}

  return (
    <div className="border-b border-(--ui-stroke-secondary) px-3 py-2">
      <div className="flex items-center gap-1.5">
        <span
          className={cn(
            'inline-block size-2 rounded-full',
            flame.ready ? 'bg-emerald-500' : flame.loaded ? 'bg-amber-500' : 'bg-(--ui-text-quaternary)'
          )}
        />
        <span className="truncate text-xs font-medium text-foreground">{flame.model || copy.noModel}</span>
      </div>
      <div className="mt-1 flex flex-wrap gap-1">
        {chips.map(c => (
          <Badge key={c} size="xs" variant="muted">
            {c}
          </Badge>
        ))}
      </div>
      {flame.blockers && flame.blockers.length > 0 ? (
        <div className="mt-1 text-[0.65rem] text-amber-600 dark:text-amber-300">{flame.blockers[0]}</div>
      ) : null}
    </div>
  )
}

function CurveTable({ curve }: { curve: DynoCurvePoint[] }) {
  const { t } = useI18n()
  const copy = t.modelDetails

  return (
    <table className="w-full text-[0.65rem] text-(--ui-text-secondary)">
      <thead className="text-(--ui-text-tertiary)">
        <tr className="text-left">
          <th className="py-0.5 pr-2 font-medium">{copy.ctx}</th>
          <th className="py-0.5 pr-2 font-medium">tok/s</th>
          <th className="py-0.5 pr-2 font-medium">{copy.prefill}</th>
          <th className="py-0.5 pr-2 font-medium">{copy.gpu}</th>
          <th className="py-0.5 font-medium">{copy.load}</th>
        </tr>
      </thead>
      <tbody>
        {curve.map(p => (
          <tr key={p.num_ctx}>
            <td className="py-0.5 pr-2">{fmtCtx(p.num_ctx)}</td>
            <td className="py-0.5 pr-2">
              {p.tok_s.toFixed(0)}
              {p.tok_s_min != null && p.tok_s_max != null ? (
                <span className="text-(--ui-text-quaternary)">
                  {' '}
                  ({p.tok_s_min.toFixed(0)}–{p.tok_s_max.toFixed(0)})
                </span>
              ) : null}
            </td>
            <td className="py-0.5 pr-2">{p.prompt_tok_s != null ? p.prompt_tok_s.toFixed(0) : '—'}</td>
            <td className="py-0.5 pr-2">{pct(p.gpu_frac)}</td>
            <td className="py-0.5">{p.load_s != null ? `${p.load_s.toFixed(1)}s` : '—'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function ModelRow({
  row,
  active,
  onUse
}: {
  row: DynoModelRow
  active: boolean
  onUse: (model: string) => void
}) {
  const { t } = useI18n()
  const copy = t.modelDetails
  const [open, setOpen] = useState(false)
  const curve = row.profile?.curve ?? []

  return (
    <div className="border-b border-(--ui-stroke-secondary)/60">
      <div className="flex items-start gap-1.5 px-3 py-1.5">
        <button
          aria-expanded={open}
          aria-label={open ? copy.collapse : copy.expand}
          className="mt-0.5 shrink-0 text-(--ui-text-tertiary) hover:text-foreground"
          disabled={!row.benched}
          onClick={() => setOpen(o => !o)}
          type="button"
        >
          {open ? <ChevronDown className="size-3" /> : <ChevronRight className={cn('size-3', !row.benched && 'opacity-30')} />}
        </button>

        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5">
            <span className={cn('truncate text-xs', active ? 'font-semibold text-foreground' : 'text-foreground')}>
              {row.model}
            </span>
            {active ? (
              <Badge size="xs" variant="default">
                {copy.active}
              </Badge>
            ) : null}
          </div>

          <div className="mt-0.5 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[0.65rem] text-(--ui-text-tertiary)">
            <span title={copy.throughputAt64k}>
              64K: <span className="text-(--ui-text-secondary)">{fmtTokS(row.kpi.tok_s_at_64k)}</span>
            </span>
            <span title={copy.operatingCtx}>
              {copy.ctx}: <span className="text-(--ui-text-secondary)">{fmtCtx(row.operating_num_ctx)}</span>
            </span>
            {row.kpi.fully_fits != null ? (
              <span className={row.kpi.fully_fits ? 'text-emerald-600 dark:text-emerald-400' : 'text-amber-600 dark:text-amber-300'}>
                {row.kpi.fully_fits ? copy.fitsGpu : copy.spillsGpu}
              </span>
            ) : null}
            {!row.benched ? <span className="text-(--ui-text-quaternary)">{copy.notBenched}</span> : null}
          </div>

          {/* Specialties */}
          {row.specialties.length > 0 ? (
            <div className="mt-1 flex flex-wrap gap-1">
              {row.specialties.map(s => (
                <Badge key={s} size="xs" variant="outline">
                  {s}
                </Badge>
              ))}
            </div>
          ) : null}

          {/* Live observed throughput — the real-world companion to the bench. */}
          {row.live ? (
            <div className="mt-1 flex items-center gap-1.5 text-[0.65rem] text-(--ui-text-tertiary)">
              <span title={copy.liveHint}>
                {copy.live}: <span className="text-(--ui-text-secondary)">{fmtTokS(row.live.recent_tok_s_avg)}</span>
                <span className="text-(--ui-text-quaternary)">
                  {' '}
                  ({row.live.min.toFixed(0)}–{row.live.max.toFixed(0)}, n={row.live.samples})
                </span>
              </span>
              <Sparkline points={row.live.trend} />
              {row.kpi.tok_s_at_64k != null &&
              Math.abs(row.live.recent_tok_s_avg - row.kpi.tok_s_at_64k) / row.kpi.tok_s_at_64k > 0.2 ? (
                <span className="text-amber-600 dark:text-amber-300" title={copy.divergesHint}>
                  {copy.diverges}
                </span>
              ) : null}
            </div>
          ) : null}
        </div>

        {!active ? (
          <button
            className="shrink-0 rounded-md px-1.5 py-0.5 text-[0.65rem] text-(--ui-text-tertiary) hover:bg-accent/50 hover:text-foreground"
            onClick={() => onUse(row.model)}
            type="button"
          >
            {copy.use}
          </button>
        ) : null}
      </div>

      {open && row.benched ? (
        <div className="bg-(--ui-surface-sunken)/40 px-3 pb-2 pl-8">
          {curve.length > 0 ? <CurveTable curve={curve} /> : null}
          <div className="mt-1.5 space-y-0.5 text-[0.6rem] text-(--ui-text-quaternary)">
            {row.profile?.gpu?.name ? (
              <div>
                {row.profile.gpu.name}
                {row.profile.gpu.vram_gb ? ` · ${row.profile.gpu.vram_gb} GB` : ''}
              </div>
            ) : null}
            {row.profile?.date ? <div>{copy.benched} {row.profile.date}</div> : null}
            {row.profile?.server_env?.ollama_version ? <div>ollama {row.profile.server_env.ollama_version}</div> : null}
            {row.chosen_because ? <div className="italic">{row.chosen_because}</div> : null}
          </div>
        </div>
      ) : null}
    </div>
  )
}

/** The model-details rail: KPI table + live throughput + running-process header. */
export function ModelDetailsPanel() {
  const { t } = useI18n()
  const copy = t.modelDetails
  const currentModel = useStore($currentModel)
  const currentProvider = useStore($currentProvider)
  const queryClient = useQueryClient()

  const query = useQuery({
    queryKey: ['model-dyno', currentModel || 'default'],
    queryFn: (): Promise<DynoReport> => getModelDyno(currentModel || undefined),
    refetchInterval: REFRESH_MS
  })

  const onUse = useMemo(
    () => (model: string) => {
      void setGlobalModel(currentProvider || '', model)
        .then(() => {
          $currentModel.set(model)
          void queryClient.invalidateQueries({ queryKey: ['model-dyno'] })
        })
        .catch(() => undefined)
    },
    [currentProvider, queryClient]
  )

  const report = query.data

  return (
    <aside className="flex h-full w-full flex-col overflow-hidden border-l border-(--ui-stroke-secondary) bg-(--ui-sidebar-surface-background)">
      <div className="flex items-center justify-between px-3 py-2">
        <span className="text-[0.8125rem] font-medium text-foreground">{copy.title}</span>
        {query.isFetching ? <GlyphSpinner className="text-xs text-(--ui-text-tertiary)" /> : null}
      </div>

      {report ? <RunningProcess report={report} /> : null}

      <div className="min-h-0 flex-1 overflow-y-auto">
        {!report && query.isPending ? (
          <div className="grid h-24 place-items-center">
            <GlyphSpinner className="text-sm text-(--ui-text-tertiary)" />
          </div>
        ) : report && report.models.length === 0 ? (
          <div className="px-3 py-6 text-center text-xs text-muted-foreground">
            {report.reachable ? copy.noModels : copy.unreachable}
          </div>
        ) : (
          report?.models.map(row => (
            <ModelRow active={row.model === currentModel} key={row.model} onUse={onUse} row={row} />
          ))
        )}
      </div>
    </aside>
  )
}
