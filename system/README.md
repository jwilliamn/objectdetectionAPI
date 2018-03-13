## Create a Gunicorn systemd Service File
Create and open a systemd service file for Gunicorn with sudo privileges in your text editor:

{% highlight bash %}
$ sudo nano /etc/systemd/system/gunicorn.service
{% endhighlight %}

Then start the Gunicorn service we created and enable it so that it starts at boot:
{% highlight bash %}
$ sudo systemctl start gunicorn
$ sudo systemctl enable gunicorn
{% endhighlight %}

We can confirm that the operation was successful by checking for the socket file.

## Check for the Gunicorn Socket File
Check the status of the process to find out whether it was able to start
{% highlight bash %}
$ sudo systemctl status gunicorn
{% endhighlight %}

If the systemctl status command indicated that an error occurred or if you do not find the myproject.sock file in the directory, it's an indication that Gunicorn was not able to start correctly. Check the Gunicorn process logs by typing:
{% highlight bash %}
$ sudo journalctl -u gunicorn
{% endhighlight %}

## Configure Nginx to Proxy Pass to Gunicorn
Now that Gunicorn is set up, we need to configure Nginx to pass traffic to the process.

Start by creating and opening a new server block in Nginx's sites-available directory:
{% highlight bash %}
$ sudo nano /etc/nginx/sites-available/object.conf
{% endhighlight %}

Save and close the file when you are finished. Now, we can enable the file by linking it to the sites-enabled directory:
{% highlight bash %}
$ sudo ln -s /etc/nginx/sites-available/myproject /etc/nginx/sites-enabled
{% endhighlight %}

Test your Nginx configuration for syntax errors by typing:

sudo nginx -t
If no errors are reported, go ahead and restart Nginx by typing:

sudo systemctl restart nginx
