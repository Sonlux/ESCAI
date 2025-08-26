{{/*
Expand the name of the chart.
*/}}
{{- define "escai.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "escai.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "escai.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "escai.labels" -}}
helm.sh/chart: {{ include "escai.chart" . }}
{{ include "escai.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "escai.selectorLabels" -}}
app.kubernetes.io/name: {{ include "escai.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "escai.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "escai.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Database URL construction
*/}}
{{- define "escai.databaseUrl" -}}
{{- if .Values.postgresql.enabled }}
{{- printf "postgresql://%s:%s@%s-postgresql:5432/%s" .Values.postgresql.auth.username .Values.postgresql.auth.password .Release.Name .Values.postgresql.auth.database }}
{{- else }}
{{- .Values.externalDatabase.url }}
{{- end }}
{{- end }}

{{/*
MongoDB URL construction
*/}}
{{- define "escai.mongodbUrl" -}}
{{- if .Values.mongodb.enabled }}
{{- printf "mongodb://%s:%s@%s-mongodb:27017/%s" .Values.mongodb.auth.username .Values.mongodb.auth.password .Release.Name .Values.mongodb.auth.database }}
{{- else }}
{{- .Values.externalMongodb.url }}
{{- end }}
{{- end }}

{{/*
Redis URL construction
*/}}
{{- define "escai.redisUrl" -}}
{{- if .Values.redis.enabled }}
{{- if .Values.redis.auth.enabled }}
{{- printf "redis://:%s@%s-redis-master:6379/0" .Values.redis.auth.password .Release.Name }}
{{- else }}
{{- printf "redis://%s-redis-master:6379/0" .Release.Name }}
{{- end }}
{{- else }}
{{- .Values.externalRedis.url }}
{{- end }}
{{- end }}

{{/*
InfluxDB URL construction
*/}}
{{- define "escai.influxdbUrl" -}}
{{- if .Values.influxdb.enabled }}
{{- printf "http://%s-influxdb:8086" .Release.Name }}
{{- else }}
{{- .Values.externalInfluxdb.url }}
{{- end }}
{{- end }}

{{/*
Neo4j URL construction
*/}}
{{- define "escai.neo4jUrl" -}}
{{- if .Values.neo4j.enabled }}
{{- printf "bolt://%s-neo4j:7687" .Release.Name }}
{{- else }}
{{- .Values.externalNeo4j.url }}
{{- end }}
{{- end }}