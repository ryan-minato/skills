#!/usr/bin/env ruby
# frozen_string_literal: true

require 'bundler/inline'

gemfile do
  source 'https://rubygems.org'
  # Add gems here. Always pin versions.
  # gem 'nokogiri', '~> 1.16'
  # gem 'httparty', '= 0.22.0'
end

require 'json'
require 'optparse'

options = {}
OptionParser.new do |opts|
  opts.banner = "Usage: ruby scripts/[name].rb [options]"
  # Add options here.
  # opts.on("--flag VALUE", "Description") { |v| options[:flag] = v }
  opts.on("-h", "--help", "Show this help message") do
    puts opts
    exit 0
  end
end.parse!

# --- implementation ---

# Data → stdout (JSON preferred):  puts JSON.pretty_generate(result)
# Diagnostics → stderr:            warn "message"
# Exit codes: exit(0) success, exit(1) error, exit(2) bad args.
