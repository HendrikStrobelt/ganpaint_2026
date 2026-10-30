<script lang="ts">
  import { Button, ContentSwitcher, RadioTile, Select, SelectItem, Switch, TileGroup } from 'carbon-components-svelte';
  import Download from 'carbon-icons-svelte/lib/Download.svelte';
  import Reset from 'carbon-icons-svelte/lib/Reset.svelte';
  import Undo from 'carbon-icons-svelte/lib/Undo.svelte';
  import type { Editor } from './state.svelte';

  let { editor }: { editor: Editor } = $props();

  let modeIndex = $state(0);
  $effect(() => {
    editor.mode = modeIndex === 0 ? 'draw' : 'remove';
  });
</script>

<section class="controls" aria-label="Painting tools">
  <div class="mode" class:remove={modeIndex === 1}>
    <ContentSwitcher bind:selectedIndex={modeIndex}>
      <Switch text="Draw" />
      <Switch text="Remove" />
    </ContentSwitcher>
  </div>

  <TileGroup legendText="Brush" bind:selected={editor.brush}>
    <div class="brushes">
      {#each editor.info?.brushes ?? [] as b (b.name)}
        <RadioTile value={b.name}>{b.label}</RadioTile>
      {/each}
    </div>
  </TileGroup>

  <Select labelText="Strength" bind:selected={editor.strength}>
    {#each Object.entries(editor.info?.strengths ?? {}) as [name, units] (name)}
      <SelectItem value={name} text={`${name} (${units} units)`} />
    {/each}
  </Select>

  <div class="actions">
    <Button kind="secondary" size="small" icon={Undo} disabled={!editor.strokes.length} onclick={() => editor.undo()}>
      Undo
    </Button>
    <Button kind="danger-tertiary" size="small" icon={Reset} disabled={!editor.strokes.length} onclick={() => editor.reset()}>
      Reset
    </Button>
    <Button
      kind="primary"
      size="small"
      icon={Download}
      href={editor.image}
      download={`ganpaint-${editor.sampleId}.${editor.image.startsWith('data:image/webp') || editor.image.endsWith('.webp') ? 'webp' : 'png'}`}
    >
      Download
    </Button>
  </div>
</section>

<style>
  .controls {
    display: flex;
    flex-direction: column;
    gap: 20px;
    width: min(100%, 512px);
  }
  .brushes {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 4px;
    margin-top: 8px;
  }
  .brushes :global(.bx--tile--selectable) {
    width: 100%;
    min-height: 0;
    margin: 0;
    padding: 10px 12px;
    text-align: center;
    text-transform: capitalize;
  }
  .brushes :global(.bx--tile__checkmark) {
    display: none;
  }
  .actions {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }
  .mode :global(.bx--content-switcher--selected) {
    color: #161616;
  }
  .mode :global(.bx--content-switcher--selected::after) {
    background: var(--draw);
  }
  .mode.remove :global(.bx--content-switcher--selected::after) {
    background: var(--remove);
  }
</style>
