from django.db import models
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin
from django.core.validators import RegexValidator
from django.conf import settings
from django.utils import timezone


REGEX_MOBILE = RegexValidator(
    r"^(?!0|1|2|3|4|5)(\d)(?!\1+$)\d{9}$", "Invalid mobile number"
)
REGEX_PINCODE = RegexValidator(r"^[1-9][0-9]{5}$", "Invalid pincode")


class TimestampModel(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class UserManager(BaseUserManager):
    def create_user(self, username, password=None, email=None):
        if CustomUser.objects.filter(username=username).exists():
            raise ValueError("Username already exists")

        user = self.model(
            username=username,
            email=self.normalize_email(email),
        )
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, username, password, email):
        user = self.create_user(
            username=username,
            email=self.normalize_email(email),
            password=password,
        )
        user.is_staff = True
        user.is_verified = True
        user.is_admin = True
        user.is_superuser = True
        user.save(using=self._db)
        return user


class Plan(TimestampModel):
    plan_name = models.CharField(max_length=50, unique=True)

    def __str__(self):
        return self.plan_name


class SubscriptionPlan(TimestampModel):
    plan = models.ForeignKey(Plan, on_delete=models.CASCADE, related_name="subscription_plans")
    price = models.DecimalField(max_digits=10, decimal_places=2)
    duration_in_days = models.IntegerField()  # e.g., 30 days, 365 days
    features = models.TextField()  # Description of features included
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return f"{self.plan.plan_name} - ${self.price}"


class AllowedPermission(TimestampModel):
    edit_profile = models.BooleanField(default=False)
    delete_profile = models.BooleanField(default=False)
    view_activity = models.BooleanField(default=False)
    delete_activity = models.BooleanField(default=False)

    # Allow campaign permissions
    personal_loan = models.BooleanField(default=False)
    insurance = models.BooleanField(default=False)


class CustomUser(AbstractBaseUser, PermissionsMixin):
    username = models.CharField(default="", max_length=50, unique=True)
    email = models.EmailField(max_length=50, default="", null=True, blank=True)
    password = models.CharField(default="", max_length=256)

    ROLE_CHOICES = [
        ("admin", "Admin"),
        ("user", "User"),  # A general user, regardless of subscription type
    ]
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, default="user")

    # Subscription Plan Information
    subscription_plan = models.ForeignKey(SubscriptionPlan, on_delete=models.SET_NULL, null=True, blank=True, related_name="users")
    subscription_start_date = models.DateTimeField(null=True, blank=True)
    subscription_end_date = models.DateTimeField(null=True, blank=True)

    # Subscription classification (could be inferred from subscription_plan)
    is_premium = models.BooleanField(default=False)

    # Other fields
    is_verified = models.BooleanField(default=False)
    is_admin = models.BooleanField(default=False)  # Admin flag
    is_staff = models.BooleanField(default=False)  # Required for Django admin
    is_active = models.BooleanField(default=True)  # Active status

    objects = UserManager()

    USERNAME_FIELD = "username"
    REQUIRED_FIELDS = ["email"]

    def has_perm(self, perm, obj=None):
        return self.is_admin

    def has_module_perms(self, app_label):
        return True

    @property
    def is_free_user(self):
        """Returns True if the user has a free subscription plan."""
        return self.subscription_plan and self.subscription_plan.plan.plan_name == "free"

    @property
    def is_basic_user(self):
        """Returns True if the user has a basic subscription plan."""
        return self.subscription_plan and self.subscription_plan.plan.plan_name == "basic"

    @property
    def is_premium_user(self):
        """Returns True if the user has a premium subscription plan."""
        return self.subscription_plan and self.subscription_plan.plan.plan_name == "premium"

    def is_subscription_active(self):
        """Returns True if the user's subscription is currently active."""
        if self.subscription_end_date:
            return timezone.now() <= self.subscription_end_date
        return False


class UserProfile(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="profile")
    full_name = models.CharField(max_length=100, blank=True, null=True)
    profile_pic = models.ImageField(upload_to='profile_pics/', null=True, blank=True)
    bio = models.TextField(max_length=500, blank=True, null=True)
    phone_number = models.CharField(max_length=15, blank=True, null=True, validators=[REGEX_MOBILE])
    address = models.CharField(max_length=255, blank=True, null=True)
    city = models.CharField(max_length=100, blank=True, null=True)
    country = models.CharField(max_length=100, blank=True, null=True)

    def __str__(self):
        return f"Profile of {self.user.username}"


class ProductType(TimestampModel):
    name = models.CharField(max_length=50)
    description = models.CharField(max_length=250, default='')

    def __str__(self):
        return self.name


class Product(TimestampModel):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="products"
    )
    title = models.CharField(max_length=100, default="", blank=True)
    description = models.TextField(blank=True, null=True)
    product_type = models.ForeignKey(ProductType, on_delete=models.SET_NULL, null=True, blank=True, related_name="products")

    def __str__(self):
        return f"Product {self.id} - {self.title} by {self.user.username}"


class ProductImage(TimestampModel):
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name="images")
    image = models.ImageField(upload_to="user_products/images/", blank=True, null=True)

    def __str__(self):
        return f"Image {self.id} for Product {self.product.title}"


class Notifications(TimestampModel):
    user = models.ForeignKey(
        CustomUser, on_delete=models.CASCADE, related_name="notifications"
    )
    title = models.CharField(max_length=150, default="")
    description = models.TextField(max_length=2000, default="")
    notification_time = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"Notification for {self.user.username}"


class UserWebDetails(TimestampModel):
    user = models.OneToOneField(CustomUser, on_delete=models.CASCADE, related_name='web_details')
    ip_addresses = models.JSONField(default=list)  # Store IPs in JSON format
    browser_info = models.CharField(max_length=200, blank=True, null=True)
    os_info = models.CharField(max_length=100, blank=True, null=True)

    def __str__(self):
        return f"Web details for {self.user.username}"


class SubscriptionPurchaseHistory(TimestampModel):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='subscription_history')
    subscription_plan = models.ForeignKey(SubscriptionPlan, on_delete=models.CASCADE)
    purchase_date = models.DateTimeField(auto_now_add=True)
    amount_paid = models.DecimalField(max_digits=10, decimal_places=2)
    payment_method = models.CharField(max_length=50)  # e.g., Credit Card, PayPal

    def __str__(self):
        return f"{self.user.username} purchased {self.subscription_plan.plan.plan_name} on {self.purchase_date}"
