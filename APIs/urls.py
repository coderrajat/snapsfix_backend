
from django.urls import path
from .views import *


urlpatterns = [
  path('remove-background/', RemoveBackgroundView.as_view(), name='remove_background'),
  path('add-background/', AddBackgroundView.as_view(), name='add_background'),
  path('add-bg-color/', AddColorBackgroundView.as_view(), name='background_color'),
  path('enhance-image/', EnhanceImageView.as_view(), name='background_color'),
]
