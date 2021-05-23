self.__BUILD_MANIFEST = {
  __rewrites: { beforeFiles: [], afterFiles: [], fallback: [] },
  '/': ['static\u002Fchunks\u002Fpages\u002Findex.js'],
  '/_error': ['static\u002Fchunks\u002Fpages\u002F_error.js'],
  '/about': ['static\u002Fchunks\u002Fpages\u002Fabout.js'],
  '/blog': ['static\u002Fchunks\u002Fpages\u002Fblog.js'],
  '/blog/[...slug]': ['static\u002Fchunks\u002Fpages\u002Fblog\u002F[...slug].js'],
  '/next/dist/pages/_error': [
    'static\u002Fchunks\u002Fpages\u002Fnext\u002Fdist\u002Fpages\u002F_error.js',
  ],
  '/projects': ['static\u002Fchunks\u002Fpages\u002Fprojects.js'],
  sortedPages: [
    '\u002F',
    '\u002F_app',
    '\u002F_error',
    '\u002Fabout',
    '\u002Fblog',
    '\u002Fblog\u002F[...slug]',
    '\u002Fnext\u002Fdist\u002Fpages\u002F_error',
    '\u002Fprojects',
  ],
}
self.__BUILD_MANIFEST_CB && self.__BUILD_MANIFEST_CB()
