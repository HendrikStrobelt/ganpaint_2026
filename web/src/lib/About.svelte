<script lang="ts">
  import { CodeSnippet, Column, Grid, Link, Row, Tile } from 'carbon-components-svelte';
  import { BIBTEX, LINKS, PAPER } from './about';
</script>

<section class="about" aria-label="About GANPaint">
  <Grid noGutter>
    <Row>
      <Column sm={4} md={4} lg={8}>
        <Tile>
          <h2>The paper</h2>
          <p class="title">{PAPER.title}</p>
          <p class="meta">{PAPER.venue}</p>
          <p>{PAPER.authors.join(', ')}</p>
          <p class="meta">{PAPER.affiliations.join(' · ')}</p>
          <p class="links">
            {#each LINKS as l (l.href)}
              <Link href={l.href} target="_blank" rel="noopener noreferrer">{l.text}</Link>
            {/each}
          </p>
          <CodeSnippet type="multi" wrapText expanded code={BIBTEX} />
        </Tile>
      </Column>
      <Column sm={4} md={4} lg={8}>
        <Tile>
          <h2>How it works</h2>
          <p>
            GAN Dissection finds units in a progressive GAN (trained on LSUN churches) whose activations line up with
            concepts found by a scene-segmentation network: trees, grass, doors, sky and more.
          </p>
          <p>
            When you paint, the units for the chosen brush are set at <code>layer4</code> inside the painted cells of
            the 32×32 feature map. <em>Draw</em> turns them on and <em>Remove</em> switches them off. The generator then
            re-renders the image, and strength sets how many units are changed.
          </p>
        </Tile>
      </Column>
    </Row>
  </Grid>
</section>

<style>
  .about {
    margin-top: 48px;
  }
  .about :global(.bx--tile) {
    height: 100%;
    margin-bottom: 1px;
  }
  h2 {
    margin: 0 0 12px;
    font-size: 1.25rem;
    font-weight: 400;
  }
  p {
    margin: 0 0 8px;
  }
  .title {
    font-weight: 600;
  }
  .meta {
    color: var(--muted);
    font-size: 0.875rem;
  }
  .links {
    display: flex;
    gap: 16px;
    margin: 16px 0;
  }
  code {
    font-family: 'IBM Plex Mono', monospace;
  }
</style>
