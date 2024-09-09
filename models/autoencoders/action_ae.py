import torch

class Encoder(torch.nn.Module):

    def __init__(self, action_dim, hidden_dim, latent_dim):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(action_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, latent_dim),
            torch.nn.Tanh(),
        )

    def forward(self, action):
        return self.model(torch.nn.functional.normalize(action, p=1, dim=1))


class Decoder(torch.nn.Module):

    def __init__(self, action_dim, hidden_dim, latent_dim):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, action_dim),
            torch.nn.Softmax(dim=1),
        )

    def forward(self, latent):
        return self.model(latent)


class AutoEncoder(torch.nn.Module):

    def __init__(self, action_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(action_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(action_dim, hidden_dim, latent_dim)

    def forward(self, action):
        return self.decoder(self.encoder(action))


class VariationalEncoder(torch.nn.Module):
    def __init__(self, action_dim, hidden_dim, latent_dim):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(action_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
        )
        self.mean = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, latent_dim)
        )
        self.log_var = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, action):
        h_ = self.model(torch.nn.functional.normalize(action, p=1, dim=1))
        return self.mean(h_), self.log_var(h_)


class VariationalDecoder(torch.nn.Module):
    def __init__(self, action_dim, hidden_dim, latent_dim):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, action_dim),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, latent):
        return self.model(latent)


class VariationalAutoEncoder(torch.nn.Module):
    def __init__(self, action_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = VariationalEncoder(action_dim, hidden_dim, latent_dim)
        self.decoder = VariationalDecoder(action_dim, hidden_dim, latent_dim)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, action):
        mean, log_var = self.encoder(action)
        params = self.reparameterize(mean, log_var)
        return self.decoder(params), params, mean, log_var


class ConditionalEncoder(torch.nn.Module):

    def __init__(self, observation_dim, action_dim, hidden_dim, latent_dim):
        super().__init__()

        self.observation_encoder = torch.nn.Sequential(
            torch.nn.Linear(observation_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
        )

        self.action_encoder = torch.nn.Sequential(
            torch.nn.Linear(action_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
        )

        self.model = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, latent_dim),
            torch.nn.Tanh(),
        )

    def forward(self, observation, action):
        action_h = self.action_encoder(torch.nn.functional.normalize(action, p=1, dim=1))
        observation_h = self.observation_encoder(observation)
        return self.model(torch.cat([action_h, observation_h], dim=1))


class ConditionalDecoder(torch.nn.Module):

    def __init__(self, observation_dim, action_dim, hidden_dim, latent_dim):
        super().__init__()

        self.observation_decoder = torch.nn.Sequential(
            torch.nn.Linear(observation_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
        )
        self.action_decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
        )

        self.model = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, action_dim),
            torch.nn.Softmax(dim=1),
        )

    def forward(self, observation, latent):
        observation_h = self.observation_decoder(observation)
        action_h = self.action_decoder(latent)
        return self.model(torch.cat([observation_h, action_h], dim=1))


class ConditionalAutoEncoder(torch.nn.Module):

    def __init__(self, observation_dim, action_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = ConditionalEncoder(observation_dim, action_dim, hidden_dim, latent_dim)
        self.decoder = ConditionalDecoder(observation_dim, action_dim, hidden_dim, latent_dim)

    def forward(self, observation, action):
        return self.decoder.forward(observation, self.encoder.forward(observation, action))
