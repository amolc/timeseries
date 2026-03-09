# Sitemap - QuantLabs

This document outlines the structure of the QuantLabs web application, detailing the pages accessible from the home page and their corresponding Django view functions.

## Navigation Structure

- **[Home Page](http://localhost:9900/)** (/)
  - **Asset Intelligence Dashboards**
    - [BTCUSD Intelligence](http://localhost:9900/dashboard/btc/) (/dashboard/btc/)
    - [GOLD Intelligence](http://localhost:9900/dashboard/gold/) (/dashboard/gold/)
    - [SPX500 Intelligence](http://localhost:9900/dashboard/spx500/) (/dashboard/spx500/)
    - [NIFTY Intelligence](http://localhost:9900/dashboard/nifty/) (/dashboard/nifty/)
  - **Main Navigation Links**
    - [Live Dashboard](http://localhost:9900/dashboard/) (/dashboard/)
      - [Drift Monitoring](http://localhost:9900/dashboard/drift-monitoring/) (/dashboard/drift-monitoring/)
      - Run Pipeline (/dashboard/run-pipeline/)
    - [A/B Testing](http://localhost:9900/ab-testing/) (/ab-testing/)
    - [ROI Analysis](http://localhost:9900/roi/) (/roi/)
    - [Blogs & Insights](http://localhost:9900/blogs/) (/blogs/)
      - [Drift Detection Insight](http://localhost:9900/blogs/drift-detection-insight/) (/blogs/drift-detection-insight/)

---

## Technical Mapping (URLs to Views)

Below is the mapping of each page to its corresponding view function in the codebase.

| Page Name | URL Path | View Function |
| :--- | :--- | :--- |
| **Home Page** | `/` | `homepage.views.landing_page` |
| **BTCUSD Dashboard** | `/dashboard/btc/` | `homepage.views.asset_dashboard` |
| **GOLD Dashboard** | `/dashboard/gold/` | `homepage.views.asset_dashboard` |
| **SPX500 Dashboard** | `/dashboard/spx500/` | `homepage.views.asset_dashboard` |
| **NIFTY Dashboard** | `/dashboard/nifty/` | `homepage.views.asset_dashboard` |
| **Live Dashboard** | `/dashboard/` | `monitoring.views.dashboard_overview` |
| **Drift Monitoring** | `/dashboard/drift-monitoring/` | `monitoring.views.drift_monitoring` |
| **Run Pipeline** | `/dashboard/run-pipeline/` | `monitoring.views.run_pipeline_view` |
| **A/B Testing** | `/ab-testing/` | `ab_testing.views.ab_testing_index` |
| **ROI Analysis** | `/roi/` | `roi.views.roi_index` |
| **Blog List** | `/blogs/` | `blogs.views.blog_list` |
| **Blog: Drift Detection** | `/blogs/drift-detection-insight/` | `blogs.views.drift_detection_blog` |
| **Admin** | `/admin/` | `django.contrib.admin.site.urls` |
