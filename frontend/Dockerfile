# SpaceNetra Frontend Nginx Container Dockerfile
FROM nginx:alpine

# Copy static frontend UI files from app/static
COPY app/static /usr/share/nginx/html

# Copy Nginx reverse proxy configuration
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
