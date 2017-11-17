
import React, { Component } from 'react';
import { render } from 'react-dom';
import Bloodhound from 'bloodhound-js';

import 'bootstrap/dist/css/bootstrap.min.css';
import 'font-awesome/css/font-awesome.min.css';
import './styles.css';

const urlTokenizer = (datum) => {
  return datum.split(/[^a-zA-Z0-9]/).filter(v => v);
};

export class NewTabPage extends Component {
  constructor(props) {
    super(props);

    this.bloodhound = new Bloodhound({
      queryTokenizer: urlTokenizer,
      initialize: false,
      identify: datum => datum.id,
      datumTokenizer: datum => urlTokenizer(datum.url),
      remote: {
        url: '/getSuggestions',
        prepare(query, req) {
          req.data = {
            type: 'QUERY_SUGGESTIONS',
            query,
          };
          return req;
        },
      }
    });

    // Setting the transport on the remote in construction doesn't work because their code is commented out... o_O
    this.bloodhound.remote.transport._send = (req, onSuccess, onError) => {
      return new Promise((resolve, reject) => {
        chrome.runtime.sendMessage(chrome.runtime.id, req.data, response => {
          if (response.err) {
            reject(response.err)
          } else if (response.data) {
            resolve(response.data);
          }
        });
      });
    }
    this.bloodhound.initialize();

    this.state = {
      currentSearch: undefined,
      suggestions: [],
    }
  }

  handleChange(e) {
    const value = e.target.value;
    this.bloodhound.initialize().then(() => {
      this.bloodhound.search(value, suggestions => {
        this.setState({
          currentSearch: value,
          suggestions,
        });
      }, asyncSuggestions => {
        this.bloodhound.add(asyncSuggestions);
        if (this.state.currentSearch == value) {
          this.setState({
            suggestions: this.state.suggestions.concat(asyncSuggestions),
          });
        }
      });
    });
  }

  handleNavigate(e) {
    e.preventDefault();
    this.onNavigate(e.target.href);
  }

  handleSubmit(e) {
    e.preventDefault();
    this.onNavigate(`https://www.google.com/search?q=${this.state.currentSearch}`);
  }

  onNavigate(url) {
    chrome.runtime.sendMessage(chrome.runtime.id, {
      type: 'NAVIGATION_COMMIT',
      url,
    }, response => {
      if (response.err) {
        console.error(response.err)
      } else if (response.data) {
        chrome.tabs.update({url, });
      }
    });
  }

  componentDidMount() {
    this.refs.input.focus();
  }

  render() {
    const suggestions = (this.state.suggestions.length > 0 ? this.state.suggestions : this.props.initialSuggestions).filter((el, i, self) => {
      return self.findIndex(other => other.id === el.id) === i;
    });
    return (
      <div style={{
        position: 'absolute',
        height: '100vh',
        width: '100%',
      }}>
          <div style={{
            position: 'relative',
            top: '25%',
          }}>
              <div className="container">
                  <div className="col-lg-8 offset-lg-2 col-sm-10 offset-sm-1">
                      <form onSubmit={this.handleSubmit.bind(this)}>
                          <div className="input-group input-group-lg">
                              <span className="input-group-addon"><i className="fa fa-search" aria-hidden="true"></i></span>
                              <input className="form-control" type="text" ref="input" style={{ width: '100%' }} onChange={this.handleChange.bind(this)} />
                          </div>
                      </form>
                      <hr />
                      <ul className="list-group">
                        {suggestions.map(suggestion => (
                          <li key={suggestion.id} className="suggestion list-group-item">
                              <a onClick={this.handleNavigate.bind(this)} href={suggestion.url}>{suggestion.url}</a>
                          </li>
                        ))}
                      </ul>
                  </div>
              </div>
          </div>
      </div>
    );
  }
}

chrome.runtime.sendMessage(chrome.runtime.id, {
  type: 'INITIAL_SUGGESTIONS',
}, response => {
  if (response.err) {
    console.error(response.err);
  } else if (response.data) {
    render(<NewTabPage initialSuggestions={response.data} />, document.body);
  }
});

render(<NewTabPage initialSuggestions={[]} />, document.body);
