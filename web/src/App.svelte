<script lang="ts">
  import { onMount } from 'svelte';
  import { Content, Header, HeaderNav, HeaderNavItem, InlineLoading, InlineNotification } from 'carbon-components-svelte';
  import About from './lib/About.svelte';
  import { LINKS, PAPER } from './lib/about';
  import Controls from './lib/Controls.svelte';
  import Gallery from './lib/Gallery.svelte';
  import History from './lib/History.svelte';
  import PaintCanvas from './lib/PaintCanvas.svelte';
  import { Editor } from './lib/state.svelte';

  const editor = new Editor();
  onMount(() => editor.init());

  function key(e: KeyboardEvent) {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z' && !(e.target instanceof HTMLSelectElement)) {
      e.preventDefault();
      editor.undo();
    }
  }
</script>

<svelte:window onkeydown={key} />

<Header companyName="MIT-IBM Watson AI Lab" platformName="GANPaint" href="#">
  <HeaderNav>
    {#each LINKS as l (l.href)}
      <HeaderNavItem href={l.href} text={l.text} target="_blank" rel="noopener noreferrer" />
    {/each}
  </HeaderNav>
</Header>

<Content>
  <main>
    <header class="intro">
      <h1>GANPaint</h1>
      <p>Paint trees, grass, doors, sky, clouds, brick and domes into a GAN-generated church.</p>
      <p class="byline">
        Based on <em>GAN Dissection</em> ({PAPER.venue}) by {PAPER.authors.join(', ')}.
      </p>
    </header>

    {#if editor.error}
      <InlineNotification kind="error" title="Error" subtitle={editor.error} hideCloseButton />
    {/if}

    {#if editor.info}
      <div class="layout">
        <PaintCanvas {editor} />
        <div class="side">
          <Controls {editor} />
          <Gallery {editor} />
          <History {editor} />
        </div>
      </div>
    {:else if !editor.error}
      <InlineLoading description="Loading…" />
    {/if}

    <About />
  </main>
</Content>

<style>
  main {
    max-width: 1100px;
    margin: 0 auto;
  }
  .intro h1 {
    margin: 0;
    font-size: 2rem;
    font-weight: 300;
  }
  .intro p {
    margin: 4px 0 0;
    color: var(--muted);
  }
  .intro {
    margin-bottom: 24px;
  }
  .layout {
    display: flex;
    flex-wrap: wrap;
    gap: 24px;
    align-items: flex-start;
  }
  .side {
    display: flex;
    flex-direction: column;
    gap: 24px;
    flex: 1 1 320px;
    min-width: 0;
  }
</style>
