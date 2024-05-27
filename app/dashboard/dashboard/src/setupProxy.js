const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function (app) {
  app.use(
    '/api',  // Specify the path you want to proxy (e.g., '/api')
    createProxyMiddleware({
      target: 'http://localhost:5000',  // Replace with your Flask server's address
      changeOrigin: true,
    })
  );
};
