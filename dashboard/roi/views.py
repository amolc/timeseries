from django.shortcuts import render
from .models import ROIMetric

def roi_index(request):
    """
    Displays simulated ROI and financial impact of the forecasting model.
    """
    latest_roi = ROIMetric.objects.order_by('-calculated_at').first()
    
    # If no data exists, create a dummy entry for demonstration
    if not latest_roi:
        latest_roi = ROIMetric.objects.create(
            model_version="1.0",
            period="Last 30 Days",
            simulated_profit_usd=12500.50,
            risk_reduction_pct=15.4,
        )

    context = {
        'latest_roi': latest_roi,
        'history': ROIMetric.objects.all().order_by('-calculated_at')[:10]
    }
    return render(request, 'dashboard/roi_index.html', context)
