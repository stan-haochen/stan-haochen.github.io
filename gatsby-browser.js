exports.onRouteUpdate = ({location}) => {
    console.log('new pathname', location.pathname);
    if (window.MathJax !== undefined) {
      window.MathJax.Hub.Queue(['Typeset', window.MathJax.Hub]);
    }
  };