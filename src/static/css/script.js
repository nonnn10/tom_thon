const options = {
  key: '0z7yC0GdxIaYJjZtEHk2C1RMqcNLAQF2', // REPLACE WITH YOUR KEY !!!
};

windyInit(options, windyAPI => {
  // All the params are stored in windyAPI.store
  const { overlays, broadcast } = windyAPI;

  const windMetric = overlays.wind.metric;
  console.log(windMetric);
  // 'kt' .. actually selected metric for wind overlay
  // Read only value! Do not modify.

  overlays.wind.listMetrics();
  // ['kt', 'bft', 'm/s', 'km/h', 'mph'] .. available metrics

  overlays.wind.setMetric('m/s');
  // Metric for wind was changed to bft

  broadcast.on('metricChanged', (overlay, newMetric) => {
    // Any changes of any metric can be observed here
    console.log(overlay, newMetric);
  });
});
