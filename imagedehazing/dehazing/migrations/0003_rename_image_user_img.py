# Generated by Django 4.0.2 on 2022-03-01 07:53

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('dehazing', '0002_rename_img_user_image'),
    ]

    operations = [
        migrations.RenameField(
            model_name='user',
            old_name='image',
            new_name='img',
        ),
    ]
