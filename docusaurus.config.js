// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const lightCodeTheme = require('prism-react-renderer/themes/github');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'The ML Engineer Notebook',
  tagline: 'A Machine Learning Engineer notebook that includes everything to become a great ML Engineer.',
  url: 'https://your-docusaurus-test-site.com',
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
          editUrl: 'https://github.com/Dipankar-Medhi/the-mlengineer-notebook',
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

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
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
            href: 'https://github.com/Dipankar-Medhi/the-mlengineer-notebook',
            label: 'GitHub',
            position: 'right',
          },
          {
            type: 'doc',
            docId: 'MLOps/intro',
            position: 'left',
            label: 'MLOps'
          },
          {
            type: 'doc',
            docId: 'system-design/system-design',
            position: 'left',
            label: 'System Design'
          },
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
                href: 'https://twitter.com/docusaurus',
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
                href: 'https://github.com/facebook/docusaurus',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} My Project, Inc. Built with Docusaurus.`,
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
      },
    }),
};

module.exports = config;
