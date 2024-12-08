# import io
# from PIL import Image
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework.parsers import MultiPartParser, FormParser
# from rest_framework import status
# from rembg import remove

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from PIL import Image,ImageEnhance
import io
from rembg import remove
from django.http import HttpResponse
import numpy as np
import cv2
import os

import base64

class RemoveBackgroundView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        try:
            image_file = request.FILES.get('image')
            if not image_file:
                return Response({'error': 'Image file is required.'}, status=status.HTTP_400_BAD_REQUEST)
            
            # Open the image
            image = Image.open(image_file)
            
            # Convert image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format=image.format)
            img_byte_arr = img_byte_arr.getvalue()
            
            # Remove background
            output = remove(img_byte_arr)
            img_removed_bg = Image.open(io.BytesIO(output)).convert("RGBA")
            
            # Save the resulting image to an output stream
            img_output_stream = io.BytesIO()
            img_removed_bg.save(img_output_stream, format='PNG')
            img_output_stream.seek(0)

            # Encode image as base64
            img_base64 = base64.b64encode(img_output_stream.getvalue()).decode('utf-8')

            # Return the base64 encoded image as JSON response
            return Response({'result_b64': img_base64}, status=status.HTTP_200_OK)
        
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)


class AddBackgroundView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        try:
            # Get the main image and background image from the request
            image_file = request.FILES['image']
            background_file = request.FILES['background']

            # Open and process the main image (remove background)
            image = Image.open(image_file)
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format=image.format)
            img_byte_arr = img_byte_arr.getvalue()
            output = remove(img_byte_arr)

            # Open the foreground image after background removal
            img_removed_bg = Image.open(io.BytesIO(output)).convert("RGBA")
            background = Image.open(background_file).convert("RGBA")
            background = background.resize(img_removed_bg.size, Image.LANCZOS)
            combined_image = Image.alpha_composite(background, img_removed_bg)
            img_output_stream = io.BytesIO()
            combined_image.save(img_output_stream, format='PNG')
            img_output_stream.seek(0)
            img_base64 = base64.b64encode(img_output_stream.getvalue()).decode('utf-8')
            return Response({'image': img_base64}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
        


class AddColorBackgroundView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        try:
            # Get the main image and color background
            image_file = request.FILES.get('image')
            color_code = request.data.get('color', '#FFFFFF')

            if not image_file:
                return Response({'error': 'Image file is required.'}, status=status.HTTP_400_BAD_REQUEST)

            # Process the main image
            image = Image.open(image_file)
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format=image.format)
            img_byte_arr = img_byte_arr.getvalue()
            output = remove(img_byte_arr)  # Perform background removal
            img_removed_bg = Image.open(io.BytesIO(output)).convert("RGBA")

            # Parse the color code
            if color_code.startswith('#'):
                color = tuple(int(color_code[i:i+2], 16) for i in (1, 3, 5)) + (255,)
            else:
                return Response({'error': 'Invalid color format.'}, status=status.HTTP_400_BAD_REQUEST)

            # Create a background image with the chosen color
            background = Image.new("RGBA", img_removed_bg.size, color)

            # Composite the images
            combined_image = Image.alpha_composite(background, img_removed_bg)

            # Convert to base64 string
            img_output_stream = io.BytesIO()
            combined_image.save(img_output_stream, format='PNG')
            img_output_stream.seek(0)
            img_base64 = base64.b64encode(img_output_stream.getvalue()).decode('utf-8')

            # Return base64 string to frontend
            return Response({'image': img_base64}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)



# class EnhanceImageView(APIView):
#     parser_classes = (MultiPartParser, FormParser)

#     def post(self, request, *args, **kwargs):
#         try:
#             # Get the image file and enhancement parameters from the request
#             image_file = request.FILES['image']
#             brightness = float(request.data.get('brightness', 1.0))  # Default to 1.0 (no change)
#             contrast = float(request.data.get('contrast', 1.0))      # Default to 1.0 (no change)
#             sharpness = float(request.data.get('sharpness', 1.0))    # Default to 1.0 (no change)

#             # Open the image
#             image = Image.open(image_file)

#             # Apply brightness enhancement
#             if brightness != 1.0:
#                 enhancer = ImageEnhance.Brightness(image)
#                 image = enhancer.enhance(brightness)

#             # Apply contrast enhancement
#             if contrast != 1.0:
#                 enhancer = ImageEnhance.Contrast(image)
#                 image = enhancer.enhance(contrast)

#             # Apply sharpness enhancement
#             if sharpness != 1.0:
#                 enhancer = ImageEnhance.Sharpness(image)
#                 image = enhancer.enhance(sharpness)

#             # Save the enhanced image to a BytesIO stream
#             img_output_stream = io.BytesIO()
#             image.save(img_output_stream, format='PNG')
#             img_output_stream.seek(0)

#             # Return the enhanced image as an HttpResponse
#             return HttpResponse(img_output_stream, content_type='image/png')

#         except Exception as e:
#             return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
        


# import requests
# from io import BytesIO
# from PIL import Image
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework.parsers import MultiPartParser, FormParser
# from super_image import EdsrModel, ImageLoader
# import torch
# import base64
# from torchvision.transforms import ToPILImage  # Import ToPILImage to handle tensor-to-image conversion

# class EnhanceImageView(APIView):
#     parser_classes = (MultiPartParser, FormParser)

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         # Load the EDSR model with scale factor 2
#         self.model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)

#     def post(self, request, *args, **kwargs):
#         try:
#             # Retrieve the uploaded image from the request
#             image_file = request.FILES.get('image')
#             if not image_file:
#                 return Response({'error': 'No image file provided.'}, status=status.HTTP_400_BAD_REQUEST)

#             # Load the image
#             image = Image.open(image_file).convert("RGB")
#             inputs = ImageLoader.load_image(image)

#             # Run the model for super-resolution
#             with torch.no_grad():
#                 preds = self.model(inputs)

#             # Convert the model output tensor to a PIL image
#             output_image = ToPILImage()(preds.squeeze(0))  # Remove batch dimension if needed

#             # Encode the upscaled image to return as base64
#             buffer = BytesIO()
#             output_image.save(buffer, format="JPEG")
#             upscaled_image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
#             return HttpResponse(output_image, content_type='image/png')

#             return Response(
#                 {
#                     'message': 'Image upscaled successfully!',
#                     'upscaled_image_base64': upscaled_image_base64
#                 },
#                 status=status.HTTP_200_OK
#             )

#         except Exception as e:
#             return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)



# from rest_framework.views import APIView
# from rest_framework.parsers import MultiPartParser, FormParser
# from super_image import EdsrModel, ImageLoader
# from PIL import Image, ImageFilter
# import torch
# from io import BytesIO
# from django.http import HttpResponse
# from torchvision.transforms import ToPILImage, ToTensor

# class EnhanceImageView(APIView):
#     parser_classes = (MultiPartParser, FormParser)

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=4)

#     def post(self, request, *args, **kwargs):
#         try:
#             image_file = request.FILES.get('image')
#             if not image_file:
#                 return Response({'error': 'No image file provided.'}, status=status.HTTP_400_BAD_REQUEST)

#             image = Image.open(image_file).convert("RGB")
#             inputs = ToTensor()(image).unsqueeze(0)

#             with torch.no_grad():
#                 preds = self.model(inputs).clamp(0, 1)

#             output_image = ToPILImage()(preds.squeeze(0)).convert("RGB")
#             output_image = output_image.filter(ImageFilter.SMOOTH)

#             buffer = BytesIO()
#             output_image.save(buffer, format="JPEG")
#             buffer.seek(0)

#             return HttpResponse(buffer, content_type='image/jpeg')

#         except Exception as e:
#             return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)



import os.path as osp
import glob2
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch


class EnhanceImageView(APIView):
    def post(self,request):
        try:
            model_path = 'models/RRDB_ESRGAN_x4.pth'  
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = arch.RRDBNet(3, 3, 64, 23, gc=32)
            model.load_state_dict(torch.load(model_path), strict=True)
            model.eval()
            model = model.to(device)

            print('Model path {:s}. \nTesting...'.format(model_path))
            path=request.FILES.get('image')
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = img * 1.0 / 255
            img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
            img_LR = img.unsqueeze(0)
            img_LR = img_LR.to(device)

            with torch.no_grad():
                output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
                output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
                output = (output * 255.0).round()

            pil_img = Image.fromarray(output)
            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG")
            buffer.seek(0)

            # Return the image as an HTTP response
            return HttpResponse(buffer, content_type="image/jpeg")

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST) 


