// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const math = require('remark-math');
const katex = require('rehype-katex');

const lightCodeTheme = require('prism-react-renderer/themes/github');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'The ML Engineer Notebook',
  tagline: 'A Machine Learning Engineer notebook that includes everything to become a great ML Engineer.',
  url: 'https://themlengineernotebook.live/',
  baseUrl: '/',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.ico',
  organizationName: 'Dipankar Medhi', // Usually your GitHub org/user name.
  projectName: 'TheMLEngineerNotebook', // Usually your repo name.

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          // Please change this to your repo.
          editUrl: 'https://github.com/Dipankar-Medhi/the-mlengineer-notebook/website/',
          remarkPlugins: [math],
          rehypePlugins: [katex],
        },
        blog: {
          showReadingTime: true,
          // Please change this to your repo.
          editUrl:
            'https://github.com/Dipankar-Medhi/the-mlengineer-notebook',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],
  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css',
      type: 'text/css',
      integrity:
        'sha384-odtC+0UGzzFL/6PNoE8rX/SPcQDXBJ+uRepguP4QkPCm2LBxH3FA3y+fKSiJ+AmM',
      crossorigin: 'anonymous',
    },
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      hideableSidebar: true,
      metadata: [{
        name: 'keywords', content: 'resources, blog, interview, questions, system design, engineer, machine learning, data science'
      }],
      navbar: {
        title: 'TheMLEngineerNotebook',
        logo: {
          alt: 'My Site Logo',
          src: 'img/docusaurus.png',
        },
        items: [
          {
            type: 'doc',
            docId: 'intro',
            position: 'left',
            label: 'Machine Learning',
          },
          {
            href: 'https://twitter.com/themlengineernb',
            className: 'navbar-icon navbar-icon-twitter',
            position: 'right',
          },
          {
            href: 'https://github.com/Dipankar-Medhi/the-mlengineer-notebook',
            className: 'navbar-icon navbar-icon-github',
            position: 'right',
          },

          { to: 'docs/ML/interview/leetcode', position: 'left', label: 'Interview' },
          { to: '/blog', label: 'Blog', position: 'left' },

        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Docs',
            items: [
              {
                label: 'Machine Learning',
                to: '/docs/intro',
              },
              {
                label: 'Interview Resources',
                to: '/docs/ML/interview/leetcode'
              },
              {
                label: 'MLOps',
                to: '/docs/MLOps/intro'
              }
            ],
          },
          {
            title: 'Community',
            items: [
              // {
              //   label: 'Discord',
              //   href: 'https://discordapp.com/invite/docusaurus',
              // },
              {
                label: 'Twitter',
                href: 'https://twitter.com/themlengineernb',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'Blog',
                to: '/blog',
              },
              {
                label: 'GitHub',
                href: 'https://github.com/Dipankar-Medhi/the-mlengineer-notebook',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} The ML Engineer Notebook. Built with ❤ and Docusaurus.`,
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
      },
    }),
};

module.exports = config;
