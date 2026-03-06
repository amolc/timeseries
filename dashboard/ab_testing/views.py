from django.shortcuts import render
from .models import ABTestRun

def ab_testing_index(request):
    """
    Displays current and past A/B tests between model versions.
    """
    active_tests = ABTestRun.objects.filter(status='Active').order_by('-start_date')
    past_tests = ABTestRun.objects.exclude(status='Active').order_by('-start_date')

    # If no data exists, create a dummy entry for demonstration
    if not active_tests and not past_tests:
        ABTestRun.objects.create(
            test_name="LSTM vs GRU Base",
            control_model_version="1.0",
            treatment_model_version="1.1",
            control_mse=0.0125,
            treatment_mse=0.0118,
            improvement_pct=5.6,
            status='Active'
        )
        active_tests = ABTestRun.objects.filter(status='Active')

    context = {
        'active_tests': active_tests,
        'past_tests': past_tests
    }
    return render(request, 'dashboard/ab_testing_index.html', context)
