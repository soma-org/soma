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
                '@fontsource-variable/literata',
                './src/styles/custom.css',
            ],
            sidebar: [
                {
                    label: 'Overview',
                    collapsed: false,
                    items: [
                        { label: 'Introduction', slug: '' },
                        { label: 'How SOMA Works', slug: 'overview/how-it-works' },
                    ],
                },
                {
                    label: 'Getting Started',
                    collapsed: false,
                    items: [
                        { label: 'Installation & Setup', slug: 'getting-started/install' },
                        { label: 'Quickstart', slug: 'getting-started/quickstart' },
                    ],
                },
                {
                    label: 'Guides',
                    collapsed: false,
                    items: [
                        { label: 'Train a Model', slug: 'guides/model-development' },
                        { label: 'Claim Rewards', slug: 'guides/claim-rewards' },
                        { label: 'Advanced Strategies', slug: 'guides/advanced-strategies' },
                        {
                            label: 'Running a Validator',
                            collapsed: true,
                            items: [
                                { label: 'Overview', slug: 'guides/validator' },
                                { label: 'Node Setup', slug: 'guides/validator/node-setup' },
                                { label: 'Join the Network', slug: 'guides/validator/join-network' },
                                { label: 'Manage a Validator', slug: 'guides/validator/manage' },
                            ],
                        },
                    ],
                },
                {
                    label: 'Concepts',
                    collapsed: false,
                    items: [
                        { label: 'Targets', slug: 'concepts/targets' },
                        { label: 'Data Submission', slug: 'concepts/submitting' },
                        { label: 'Models', slug: 'concepts/models' },
                        { label: 'Economics', slug: 'concepts/economics' },
                        { label: 'Network', slug: 'concepts/network' },
                    ],
                },
                {
                    label: 'Reference',
                    collapsed: false,
                    items: [
                        {
                            label: 'CLI',
                            collapsed: true,
                            items: [
                                { label: 'Overview', slug: 'reference/cli/overview' },
                                { label: 'Common Commands', slug: 'reference/cli/common' },
                                { label: 'Management', slug: 'reference/cli/management' },
                                { label: 'Submissions', slug: 'reference/cli/submissions' },
                                { label: 'Operators & Nodes', slug: 'reference/cli/operators' },
                                { label: 'Query Commands', slug: 'reference/cli/queries' },
                            ],
                        },
                        {
                            label: 'Python SDK',
                            collapsed: true,
                            items: [
                                { label: 'Overview', slug: 'reference/sdk/overview' },
                                { label: 'Keypair', slug: 'reference/sdk/keypair' },
                                { label: 'Chain & State', slug: 'reference/sdk/chain-state' },
                                { label: 'Targets', slug: 'reference/sdk/targets-challenges' },
                                { label: 'Transactions', slug: 'reference/sdk/transactions' },
                                { label: 'Scoring, Admin & Convenience', slug: 'reference/sdk/scoring-admin' },
                                { label: 'Types', slug: 'reference/sdk/types' },
                            ],
                        },
                        {
                            label: 'Models',
                            collapsed: true,
                            items: [
                                { label: 'Overview', slug: 'reference/models/overview' },
                            ],
                        },
                        // { label: 'Further Reading', slug: 'reference/reading' },
                        { label: 'Community', slug: 'reference/community' },
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
