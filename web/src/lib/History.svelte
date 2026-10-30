<script lang="ts">
  import { Button } from 'carbon-components-svelte';
  import TrashCan from 'carbon-icons-svelte/lib/TrashCan.svelte';
  import type { Editor } from './state.svelte';

  let { editor }: { editor: Editor } = $props();
</script>

{#if editor.strokes.length}
  <section class="history" aria-label="Strokes">
    <h2>Strokes</h2>
    <ol>
      {#each editor.strokes as s, i (s.key)}
        <li onmouseenter={() => (editor.hovered = s.key)} onmouseleave={() => (editor.hovered = null)}>
          <span class="dot {s.mode}" aria-hidden="true"></span>
          <span class="name">{i + 1}. {s.mode} {s.brush} ({s.strength})</span>
          <Button
            kind="ghost"
            size="small"
            icon={TrashCan}
            iconDescription={`Delete stroke ${i + 1}`}
            tooltipPosition="left"
            onclick={() => editor.removeStroke(s.key)}
          />
        </li>
      {/each}
    </ol>
  </section>
{/if}

<style>
  .history {
    width: min(100%, 512px);
  }
  h2 {
    margin: 0 0 8px;
    font-size: 0.875rem;
    font-weight: 400;
    color: var(--muted);
  }
  ol {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 1px;
    max-height: 220px;
    overflow-y: auto;
  }
  li {
    display: flex;
    align-items: center;
    gap: 12px;
    padding-left: 12px;
    background: var(--cds-layer-01, #262626);
    font-size: 0.875rem;
  }
  li:hover {
    background: var(--cds-layer-hover-01, #333);
    outline: 1px solid var(--cds-focus, #fff);
    outline-offset: -1px;
  }
  .name {
    flex: 1;
  }
  .dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
  }
  .dot.draw {
    background: var(--draw);
  }
  .dot.remove {
    background: var(--remove);
  }
</style>
