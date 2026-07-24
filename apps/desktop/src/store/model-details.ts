import { SIDEBAR_COLLAPSE_MEDIA_QUERY } from '@/app/layout-constants'
import { PANE_TOGGLE_REVEAL_EVENT } from '@/components/pane-shell'
import { matchesQuery } from '@/hooks/use-media-query'
import { Codecs, persistentAtom } from '@/lib/persisted'

// The "More Details" rail: a dockable right-side pane showing every local
// Dyno-benched model with its KPI stats (throughput at 64k, operating context,
// GPU fit), an expandable full power-curve spread, auto-derived specialties,
// and the live observed-throughput history that lets a bench number be checked
// against real use. Toggled from the gear beside the composer's model pill.
//
// Must match the <Pane id> registered in the contrib controller (the
// forced-reveal event is addressed by pane id).
export const MODEL_DETAILS_PANE_ID = 'model-details'

const OPEN_KEY = 'hermes.desktop.modelDetailsOpen'

// Persisted so the rail stays open across reloads, like the other rail panes.
export const $modelDetailsOpen = persistentAtom(OPEN_KEY, false, Codecs.bool)

export function openModelDetails(): void {
  $modelDetailsOpen.set(true)
}

export function closeModelDetails(): void {
  $modelDetailsOpen.set(false)
}

export function toggleModelDetails(): void {
  // Narrow width: slide the pane in as a collapsed overlay via the forced-reveal
  // pin (like the review rail under a narrow window), never the docked open
  // state which a 0px track would render invisibly.
  if (matchesQuery(SIDEBAR_COLLAPSE_MEDIA_QUERY)) {
    if (!$modelDetailsOpen.get()) {
      openModelDetails()
    }

    window.dispatchEvent(new CustomEvent(PANE_TOGGLE_REVEAL_EVENT, { detail: { id: MODEL_DETAILS_PANE_ID } }))

    return
  }

  if ($modelDetailsOpen.get()) {
    closeModelDetails()
  } else {
    openModelDetails()
  }
}
