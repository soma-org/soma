// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightPageActions from 'starlight-page-actions';
import cloudflare from '@astrojs/cloudflare';
import sitemap from '@astrojs/sitemap';

export default defineConfig({
    site: 'https://docs.soma.org',

    integrations: [
        starlight({
            title: 'SOMA',
            plugins: [
                starlightPageActions({
                    baseUrl: 'https://docs.soma.org',
                    prompt: 'Read {url}. I want to ask questions about it.',
                    actions: {
                        chatgpt: true,
                        claude: true,
                        t3chat: false,
                        v0: false,
                        markdown: true,
                    },
                    share: false,
                }),
            ],
            description: 'Documentation for the SOMA network',
            logo: {
                dark: './src/assets/wordmark_white.svg',
                light: './src/assets/wordmark_black.svg',
                replacesTitle: true,
            },
            head: [
                { tag: 'meta', attrs: { property: 'og:image', content: 'https://soma.org/thumbnail.png' } },
                { tag: 'meta', attrs: { name: 'twitter:card', content: 'summary_large_image' } },
                { tag: 'meta', attrs: { name: 'twitter:image:alt', content: 'SOMA' } },
            ],
            components: {
                Head: './src/components/Head.astro',
                Pagination: './src/components/Pagination.astro',
                TableOfContents: './src/components/TableOfContents.astro',
            },
            tableOfContents: { minHeadingLevel: 1, maxHeadingLevel: 2 },
            social: [
                { icon: 'github', label: 'GitHub', href: 'https://github.com/soma-org/soma' },
                { icon: 'x.com', label: 'X', href: 'https://x.com/soma' },
            ],
            customCss: [
                '@fontsource-variable/inter',
                '@fontsource-variable/newsreader',
                './src/styles/custom.css',
            ],
            sidebar: [
                { label: 'Introduction', slug: '' },
                { label: 'Quickstart', slug: 'quickstart' },
                {
                    label: 'Concepts',
                    collapsed: false,
                    items: [
                        { label: 'The market', slug: 'concepts/market' },
                        { label: 'The USDC bridge', slug: 'concepts/bridge' },
                    ],
                },
                {
                    label: 'Guides',
                    collapsed: false,
                    items: [
                        { label: 'Use the network', slug: 'guides/use-the-network' },
                        { label: 'Become a provider', slug: 'guides/become-a-provider' },
                    ],
                },
                {
                    label: 'Reference',
                    collapsed: false,
                    items: [
                        { label: 'CLI', slug: 'reference/cli' },
                    ],
                },
            ],
        }),
        sitemap(),
    ],

    markdown: {
        shikiConfig: {
            themes: {
                dark: 'github-dark-high-contrast',
                light: 'github-light-high-contrast',
            },
            defaultColor: 'dark',
        },
    },

    adapter: cloudflare(),
});
