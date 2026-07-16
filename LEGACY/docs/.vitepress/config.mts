import { defineConfig } from 'vitepress'

export default defineConfig({
  title: "SpectraFormer",
  description: "Transformer-based Raman spectra unmixing for graphene buffer layers on SiC substrates",

  appearance: false,

  base: '/SpectraFormer/',

  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/SpectraFormer/logo.svg' }],
  ],

  themeConfig: {
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Guide', link: '/installation' },
    ],

    sidebar: [
      {
        text: 'Getting Started',
        items: [
          { text: 'Installation', link: '/installation' },
        ]
      },
      {
        text: 'Guide',
        items: [
          { text: 'Inference', link: '/inference' },
          { text: 'Training', link: '/training' },
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/pietronvll/SpectraFormer' },
      {
        icon: {
          svg: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"><path fill="currentColor" d="M14 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V8l-6-6zM6 20V4h7v5h5v11H6z"/></svg>'
        },
        link: 'https://arxiv.org/abs/2601.04445',
        ariaLabel: 'arXiv Paper'
      }
    ],
    search: {
      provider: 'local'
    }
  }
})
